#include <engine/Optimizer.hpp>
#include <utility/Timing.hpp>
#include <utility/Logging.hpp>

using namespace Utility;

#include <sstream>
#include <iomanip>

namespace Engine
{
    Optimizer::Optimizer(std::shared_ptr<Engine::Method> method)
    {
        this->method = method;

        this->noi = this->method->systems.size();
        this->nos = this->method->systems[0]->nos;

		this->n_iterations      = this->method->parameters->n_iterations;
        this->n_iterations_log  = this->method->parameters->n_iterations_log;
        this->n_log             = this->n_iterations/this->n_iterations_log;

        // Create shared pointers to the method's systems' configurations
	    this->configurations = std::vector<std::shared_ptr<vectorfield>>(noi);
        for (int i=0; i<noi; ++i) this->configurations[i] = this->method->systems[i]->spins;
        
        // Allocate force array
        this->force = std::vector<vectorfield>(this->noi, vectorfield(this->nos, Vector3::Zero()));	// [noi][3*nos]

        // Setup Timings
        for (int i=0; i<7; ++i) this->t_iterations.push_back(system_clock::now());
        this->ips = 0;
        this->starttime = Timing::CurrentDateTime();

        // Initial force calculation s.t. it does not seem to be already converged
        this->method->Calculate_Force(this->configurations, this->force);
        // Post iteration hook to get forceMaxAbsComponent etc
        this->method->Hook_Post_Iteration();

        // Printing precision for Scalars
        #ifdef CORE_SCALAR_TYPE_FLOAT
            this->print_precision = 8;
        #else
            this->print_precision = 12;
        #endif
    }

    
    void Optimizer::Iterate()
    {
		//------------------------ Init local vars ---------------------------------
		this->starttime = Timing::CurrentDateTime();
        auto sender     = method->SenderName;
        //----
		int iteration=0, step = 0;
        //----
        std::stringstream maxforce_stream;
        maxforce_stream << std::fixed << std::setprecision(this->print_precision) << this->method->force_maxAbsComponent;
        std::string maxforce = maxforce_stream.str();
        std::stringstream force_param_stream;
        force_param_stream << std::fixed << std::setprecision(this->print_precision) << this->method->parameters->force_convergence;
        std::string force_param = force_param_stream.str();
		//------------------------ End Init ----------------------------------------

        //---- Log messages
		Log.SendBlock(Log_Level::All, sender,
			{
				"------------  Started  " + this->method->Name() + " Calculation  ------------",
				"    Going to iterate " + std::to_string(n_log) + " steps",
				"                with " + std::to_string(n_iterations_log) + " iterations per step",
				"    Force convergence parameter: " + force_param,
				"    Maximum force component:     " + maxforce,
				"    Optimizer: " + this->FullName(),
				"-----------------------------------------------------"
			}, this->method->idx_image, this->method->idx_chain);

        //---- Start Timings
		auto t_start = system_clock::now();
		auto t_current = system_clock::now();
		auto t_last = system_clock::now();

        //---- Initial save
		this->method->Save_Current(this->starttime, iteration, true, false);

        //---- Iteration loop
		for (iteration = 0; iteration < n_iterations && this->method->ContinueIterating() && !this->StopFilePresent(); ++iteration)
		{
			// Lock Systems
			this->method->Lock();

            // Pre-Iteration hook
            this->method->Hook_Pre_Iteration();
			// Do one single Iteration
			this->Iteration();
            // Post-Iteration hook
            this->method->Hook_Post_Iteration();

			// Recalculate FPS
			this->t_iterations.pop_front();
			this->t_iterations.push_back(system_clock::now());

			// Log Output every n_iterations_log steps
			if (iteration>0 && 0 == fmod(iteration, n_iterations_log))
			{
				++step;

				t_last = t_current;
				t_current = system_clock::now();

                maxforce_stream.str(std::string());
                maxforce_stream.clear();
                maxforce_stream << std::fixed << std::setprecision(this->print_precision) << this->method->force_maxAbsComponent;
                maxforce = maxforce_stream.str();

				Log.SendBlock(Log_Level::All, sender,
					{
						"----- " + this->method->Name() + " Calculation (" + this->Name() + " Optimizer): " + Timing::DateTimePassed(t_start, t_current),
						"    Step                         " + std::to_string(step) + " / " + std::to_string(n_log),
						"    Iteration                    " + std::to_string(iteration) + " / " + std::to_string(n_iterations),
						"    Time since last step:        " + Timing::DateTimePassed(t_last, t_current),
						"    Iterations / sec:            " + std::to_string(n_iterations_log / Timing::SecondsPassed(t_last, t_current)),
						"    Force convergence parameter: " + force_param,
						"    Maximum force component:     " + maxforce
					}, this->method->idx_image, this->method->idx_chain);

				this->method->Save_Current(this->starttime, iteration, false, false);

				//output_strings[step - 1] = IO::Spins_to_String(c->images[0].get());
			}

			// Unlock Systems
			this->method->Unlock();
		}// endif i

        //---- End timings
		auto t_end = system_clock::now();

        //---- Maximum force component as string
        maxforce_stream.str(std::string());
        maxforce_stream.clear();
        maxforce_stream << std::fixed << std::setprecision(this->print_precision) << this->method->force_maxAbsComponent;
        maxforce = maxforce_stream.str();

        //---- Termination Reason
        std::string reason = "";
        if (this->StopFilePresent())
			reason = "A STOP file has been found";
        else if (this->method->Force_Converged())
            reason = "The force converged";

        //---- Log messages
		std::vector<std::string> block;
		block.push_back("------------ Terminated " + this->method->Name() + " Calculation ------------");
		if (reason.length() > 0)
			block.push_back("----- Reason:   " + reason);
		block.push_back("----- Duration:       " + Timing::DateTimePassed(t_start, t_end));
		block.push_back("    Step              " + std::to_string(step) + " / " + std::to_string(n_log));
		block.push_back("    Iteration         " + std::to_string(iteration) + " / " + std::to_string(n_iterations));
        block.push_back("    Iterations / sec: " + std::to_string(iteration / Timing::SecondsPassed(t_start, t_end)));
		block.push_back("    Force convergence parameter: " + force_param);
		block.push_back("    Maximum force component:     " + maxforce);
		block.push_back("    Optimizer: " + this->FullName());
		block.push_back("-----------------------------------------------------");
		Log.SendBlock(Log_Level::All, sender, block, this->method->idx_image, this->method->idx_chain);

        //---- Final save
		this->method->Save_Current(this->starttime, iteration, false, true);
        //---- Finalize (set iterations_allowed to false etc.)
        this->method->Finalize();
    }

    
    void Optimizer::Iteration()
    {
        // Not Implemented!
        Log(Log_Level::Error, Log_Sender::All, std::string("Tried to use Optimizer::Step() of the Optimizer base class!"), this->method->idx_image, this->method->idx_chain);
    }



    scalar Optimizer::getIterationsPerSecond()
    {
        scalar l_ips = 0.0;
        for (unsigned int i = 0; i < t_iterations.size() - 1; ++i)
        {
            l_ips += Timing::SecondsPassed(t_iterations[i], t_iterations[i+1]);
        }
        this->ips = 1.0 / (l_ips / (t_iterations.size() - 1));
        return this->ips;
    }

    bool Optimizer::StopFilePresent()
    {
        std::ifstream f("STOP");
        return f.good();
    }

    // Optimizer name as string
    std::string Optimizer::Name()
    {
        // Not Implemented!
        Log(Log_Level::Error, Log_Sender::All, std::string("Tried to use Optimizer::Name() of the Optimizer base class!"), this->method->idx_image, this->method->idx_chain);
        return "--";
    }

    std::string Optimizer::FullName()
    {
        // Not Implemented!
        Log(Log_Level::Error, Log_Sender::All, std::string("Tried to use Optimizer::Fullname() of the Optimizer base class!"), this->method->idx_image, this->method->idx_chain);
        return "--";
    }
}