#include <engine/Method.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Manifoldmath.hpp>
#include <utility/Logging.hpp>
#include <utility/Timing.hpp>
#include <utility/Exception.hpp>
#include <utility/Constants.hpp>

#include <sstream>
#include <iomanip>

using namespace Utility;

namespace Engine
{
    Method::Method(std::shared_ptr<Data::Parameters_Method> parameters, int idx_img, int idx_chain) :
        parameters(parameters), idx_image(idx_img), idx_chain(idx_chain)
    {
        // Sender name for log messages
        this->SenderName = Utility::Log_Sender::All;

        // Initial maximum of absolute values of force components
        this->force_max_abs_component = parameters->force_convergence + 1.0;

        // Default history contains force_max_abs_component
        this->history = std::map<std::string, std::vector<scalar>>{
            {"force_max_abs_component", {this->force_max_abs_component}} };

        // TODO: is this a good idea?
        this->n_iterations     = this->parameters->n_iterations;
        this->n_iterations_log = this->parameters->n_iterations_log;
        this->n_log            = this->n_iterations / this->n_iterations_log;

        // Setup timings
        for (int i = 0; i<7; ++i) this->t_iterations.push_back(system_clock::now());
        this->ips = 0;
        this->starttime = Timing::CurrentDateTime();


        // Printing precision for scalars
        #ifdef CORE_SCALAR_TYPE_FLOAT
            this->print_precision = 8;
        #else
            this->print_precision = 12;
        #endif

        this->xi = vectorfield(this->nos, {0,0,0});
    }


    void Method::Iterate()
    {
        std::cerr << "iterate start" << std::endl;
        //---- Initialise Solver-specific variables
        this->Solver_Initialise();
        std::cerr << "s0lver initialized" << std::endl;
        //---- Init local vars
        std::stringstream maxforce_stream;
        maxforce_stream << std::fixed << std::setprecision(this->print_precision) << this->force_max_abs_component;
        std::string maxforce = maxforce_stream.str();
        std::stringstream force_param_stream;
        force_param_stream << std::fixed << std::setprecision(this->print_precision) << this->parameters->force_convergence;
        std::string force_param = force_param_stream.str();

        //---- Log messages
        Log.SendBlock(Log_Level::All, this->SenderName,
            {
                "------------  Started  " + this->Name() + " Calculation  ------------",
                "    Going to iterate " + std::to_string(this->n_log) + " steps",
                "                with " + std::to_string(this->n_iterations_log) + " iterations per step",
                "    Force convergence parameter: " + force_param,
                "    Maximum force component:     " + maxforce,
                "    Solver: " + this->SolverFullName(),
                "-----------------------------------------------------"
            }, this->idx_image, this->idx_chain);

        //---- Start timings
        this->starttime = Timing::CurrentDateTime();
        auto t_start = system_clock::now();
        auto t_current = system_clock::now();
        auto t_last = system_clock::now();

        //---- Initial save
        this->Save_Current(this->starttime, this->iteration, true, false);

        //---- Iteration loop
        this->iteration = 0;
        this->step      = 0;
        for ( this->iteration = 0; 
              this->ContinueIterating() &&
              !this->Walltime_Expired(t_current - t_start); 
              ++this->iteration )
        {
            t_current = system_clock::now();

            // Lock Systems
            this->Lock();

            // Pre-iteration hook
            this->Hook_Pre_Iteration();
            // Do one single Iteration
            this->Solver_Iteration();
            // Post-iteration hook
            this->Hook_Post_Iteration();

            // Recalculate FPS
            this->t_iterations.pop_front();
            this->t_iterations.push_back(system_clock::now());

            // Log Output every n_iterations_log steps
            if ( this->iteration>0 && 0 == fmod( this->iteration, this->n_iterations_log))
            {
                ++step;

                t_current = system_clock::now();

                maxforce_stream.str(std::string());
                maxforce_stream.clear();
                maxforce_stream << std::fixed << std::setprecision(this->print_precision) << this->force_max_abs_component;
                maxforce = maxforce_stream.str();

                Log.SendBlock(Log_Level::All, this->SenderName,
                    {
                        "----- " + this->Name() + " Calculation (" + this->SolverName() + " Solver): " + Timing::DateTimePassed(t_current - t_start),
                        "    Step                         " + std::to_string(step) + " / " + std::to_string(n_log),
                        "    Iteration                    " + std::to_string( this->iteration) + " / " + std::to_string(n_iterations),
                        "    Time since last step:        " + Timing::DateTimePassed(t_current - t_last),
                        "    Iterations / sec:            " + std::to_string(this->n_iterations_log / Timing::SecondsPassed(t_current - t_last)),
                        "    Force convergence parameter: " + force_param,
                        "    Maximum force component:     " + maxforce
                    }, this->idx_image, this->idx_chain);

                this->Save_Current(this->starttime, this->iteration, false, false);

                t_last = t_current;
            }

            // Unlock systems
            this->Unlock();
        }

        //---- End timings
        auto t_end = system_clock::now();

        //---- Maximum force component as string
        maxforce_stream.str(std::string());
        maxforce_stream.clear();
        maxforce_stream << std::fixed << std::setprecision(this->print_precision) << this->force_max_abs_component;
        maxforce = maxforce_stream.str();

        //---- Termination reason
        std::string reason = "";
        if (this->StopFile_Present())
            reason = "A STOP file has been found";
        else if (this->Force_Converged())
            reason = "The force converged";
        else if (this->Walltime_Expired(t_end - t_start))
            reason = "The maximum walltime has been reached";

        //---- Log messages
        std::vector<std::string> block;
        block.push_back("------------ Terminated " + this->Name() + " Calculation ------------");
        if (reason.length() > 0)
            block.push_back("----- Reason:   " + reason);
        block.push_back("----- Duration:       " + Timing::DateTimePassed(t_end - t_start));
        block.push_back("    Step              " + std::to_string(step) + " / " + std::to_string(n_log));
        block.push_back("    Iteration         " + std::to_string( this->iteration) + " / " + std::to_string(n_iterations));
        block.push_back("    Iterations / sec: " + std::to_string( this->iteration / Timing::SecondsPassed(t_end - t_start)));
        block.push_back("    Force convergence parameter: " + force_param);
        block.push_back("    Maximum force component:     " + maxforce);
        block.push_back("    Solver: " + this->SolverFullName());
        block.push_back("-----------------------------------------------------");
        Log.SendBlock(Log_Level::All, this->SenderName, block, this->idx_image, this->idx_chain);

        //---- Final save
        this->Save_Current(this->starttime, this->iteration, false, true);
        //---- Finalize (set iterations_allowed to false etc.)
        this->Finalize();
    }



    scalar Method::getIterationsPerSecond()
    {
        scalar l_ips = 0.0;
        for (unsigned int i = 0; i < t_iterations.size() - 1; ++i)
        {
            l_ips += Timing::SecondsPassed(t_iterations[i+1] - t_iterations[i]);
        }
        this->ips = 1.0 / (l_ips / (t_iterations.size() - 1));
        return this->ips;
    }


    scalar Method::getForceMaxAbsComponent()
    {
        return this->force_max_abs_component;
    }


    ///////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////// Protected functions

    void Method::Solver_Initialise()
    {
        // Not Implemented!
        Log(Log_Level::Error, Log_Sender::All, std::string("Tried to use Method::Solver_Initialise() of the Method base class!"), this->idx_image, this->idx_chain);
    }

    void Method::Solver_Iteration()
    {
        // Not Implemented!
        Log(Log_Level::Error, Log_Sender::All, std::string("Tried to use Method::Solver_Iteration() of the Method base class!"), this->idx_image, this->idx_chain);
    }


    void Method::Calculate_Force(const std::vector<std::shared_ptr<vectorfield>> & configurations, std::vector<vectorfield> & forces)
    {

    }


    void Method::VirtualForce(  const vectorfield & spins,
                                Data::Parameters_Method_LLG & llg_params,
                                const vectorfield & effective_field,
                                vectorfield & force )
    {
        //========================= Init local vars ================================
        // time steps
        scalar damping = llg_params.damping;
        // dt = time_step [ps] * gyromagnetic ratio / mu_B / (1+damping^2) <- not implemented
        scalar dtg = llg_params.dt * Constants::gamma / Constants::mu_B / (1 + damping*damping);
        scalar sqrtdtg = dtg / std::sqrt( llg_params.dt );
        // STT
        scalar a_j = llg_params.stt_magnitude;
        Vector3 s_c_vec = llg_params.stt_polarisation_normal;
        //------------------------ End Init ----------------------------------------

        Vectormath::fill       (force, {0,0,0});
        Vectormath::add_c_a    (-0.5 * dtg, effective_field, force);
        Vectormath::add_c_cross(-0.5 * dtg * damping, spins, effective_field, force);

        // STT
        if (a_j > 0)
        {
            Vectormath::add_c_a    ( 0.5 * dtg * a_j * damping, s_c_vec, force);
            Vectormath::add_c_cross( 0.5 * dtg * a_j, s_c_vec, spins, force);
        }

        // Temperature
        if (llg_params.temperature > 0)
        {
            scalar epsilon = std::sqrt(2.0*llg_params.damping / (1.0 + std::pow(llg_params.damping, 2)) * llg_params.temperature * Utility::Constants::k_B);
            Vectormath::get_random_vectorfield(llg_params, xi);
            Vectormath::add_c_a    (-0.5 * sqrtdtg * epsilon, xi, force);
            Vectormath::add_c_cross(-0.5 * sqrtdtg * damping * epsilon, spins, xi, force);
        }

        // Apply Pinning
        #ifdef SPIRIT_ENABLE_PINNING
            Vectormath::set_c_a(1, force, force, llg_params.pinning->mask_unpinned);
        #endif // SPIRIT_ENABLE_PINNING
    }




    bool Method::ContinueIterating()
    {
        return  this->iteration < this->n_iterations &&
                this->Iterations_Allowed() &&
               !this->StopFile_Present() && 
               !this->Force_Converged();
    }

    bool Method::Iterations_Allowed()
    {
        return this->systems[0]->iteration_allowed;
    }

    bool Method::Force_Converged()
    {
        bool converged = false;
        if ( this->force_max_abs_component < this->parameters->force_convergence ) converged = true;
        return converged;
    }

    bool Method::Walltime_Expired(duration<scalar> dt_seconds)
    {
        if (this->parameters->max_walltime_sec <= 0)
            return false;
        else
            return dt_seconds.count() > this->parameters->max_walltime_sec;
    }

    bool Method::StopFile_Present()
    {
        std::ifstream f("STOP");
        return f.good();
    }





    void Method::Save_Current(std::string starttime, int iteration, bool initial, bool final)
    {
        // Not Implemented!
        Log(Utility::Log_Level::Error, Utility::Log_Sender::All, std::string("Tried to use Method::Save_Current() of the Method base class!"));
        throw Utility::Exception::Not_Implemented;
    }

    void Method::Hook_Pre_Iteration()
    {
        // Not Implemented!
        Log(Utility::Log_Level::Error, Utility::Log_Sender::All, std::string("Tried to use Method::Hook_Pre_Iteration() of the Method base class!"));
        throw Utility::Exception::Not_Implemented;
    }

    void Method::Hook_Post_Iteration()
    {
        // Not Implemented!
        Log(Utility::Log_Level::Error, Utility::Log_Sender::All, std::string("Tried to use Method::Hook_Post_Iteration() of the Method base class!"));
        throw Utility::Exception::Not_Implemented;
    }

    void Method::Finalize()
    {
        // Not Implemented!
        Log(Utility::Log_Level::Error, Utility::Log_Sender::All, std::string("Tried to use Method::Finalize() of the Method base class!"));
        throw Utility::Exception::Not_Implemented;
    }

    // Return the maximum of absolute values of force components for an image
    scalar  Method::Force_on_Image_MaxAbsComponent(const vectorfield & image, vectorfield & force)
    {
        // Take out component in direction of v2
        Manifoldmath::project_tangential(force, image);

        // We want the Maximum of Absolute Values of all force components on all images
        return Vectormath::max_abs_component(force);
    }





    void Method::Lock()
    {
        for (auto& system : this->systems) system->Lock();
    }

    void Method::Unlock()
    {
        for (auto& system : this->systems) system->Unlock();
    }


    std::string Method::Name()
    {
        // Not Implemented!
        Log(Utility::Log_Level::Error, Utility::Log_Sender::All, std::string("Tried to use Method::Name() of the Method base class!"));
        return "--";
    }

    // Solver name as string
    std::string Method::SolverName()
    {
        // Not Implemented!
        Log(Log_Level::Error, Log_Sender::All, std::string("Tried to use Method::SolverName() of the Method base class!"), this->idx_image, this->idx_chain);
        return "--";
    }

    std::string Method::SolverFullName()
    {
        // Not Implemented!
        Log(Log_Level::Error, Log_Sender::All, std::string("Tried to use Method::SolverFullname() of the Method base class!"), this->idx_image, this->idx_chain);
        return "--";
    }
}