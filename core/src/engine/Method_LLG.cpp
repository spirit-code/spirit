#include <Spirit_Defines.h>
#include <engine/Method_LLG.hpp>
#include <engine/Vectormath.hpp>
#include <data/Spin_System.hpp>
#include <data/Spin_System_Chain.hpp>
#include <io/IO.hpp>
#include <utility/Logging.hpp>

#include <iostream>
#include <ctime>
#include <math.h>

#include <fmt/format.h>

using namespace Utility;

namespace Engine
{
    template <Solver solver>
    Method_LLG<solver>::Method_LLG(std::shared_ptr<Data::Spin_System> system, int idx_img, int idx_chain) :
        Method_Solver<solver>(system->llg_parameters, idx_img, idx_chain)
    {
        // Currently we only support a single image being iterated at once:
        this->systems = std::vector<std::shared_ptr<Data::Spin_System>>(1, system);
        this->SenderName = Utility::Log_Sender::LLG;

        this->noi = this->systems.size();
        this->nos = this->systems[0]->nos;

        // Forces
        this->forces    = std::vector<vectorfield>(this->noi, vectorfield(this->nos));
        this->forces_virtual    = std::vector<vectorfield>(this->noi, vectorfield(this->nos));
        this->Gradient = std::vector<vectorfield>(this->noi, vectorfield(this->nos));
        this->xi = vectorfield(this->nos, {0,0,0});
        this->s_c_grad = vectorfield(this->nos, {0,0,0});

        // We assume it is not converged before the first iteration
        this->force_converged = std::vector<bool>(this->noi, false);
        this->force_max_abs_component = system->llg_parameters->force_convergence + 1.0;

        // History
        this->history = std::map<std::string, std::vector<scalar>>{
            {"max_torque_component", {this->force_max_abs_component}},
            {"E", {this->force_max_abs_component}},
            {"M_z", {this->force_max_abs_component}} };



        // Create shared pointers to the method's systems' spin configurations
        this->configurations = std::vector<std::shared_ptr<vectorfield>>(this->noi);
        for (int i = 0; i<this->noi; ++i) this->configurations[i] = this->systems[i]->spins;

        // Allocate force array
        //this->force = std::vector<vectorfield>(this->noi, vectorfield(this->nos, Vector3::Zero()));	// [noi][3*nos]

        // Initial force calculation s.t. it does not seem to be already converged
        this->Calculate_Force(this->configurations, this->forces);
        this->Calculate_Force_Virtual(this->configurations, this->forces, this->forces_virtual);
        // Post iteration hook to get forceMaxAbsComponent etc
        this->Hook_Post_Iteration();

    }


    template <Solver solver>
    void Method_LLG<solver>::Calculate_Force(const std::vector<std::shared_ptr<vectorfield>> & configurations, std::vector<vectorfield> & forces)
    {
        // Loop over images to calculate the total force on each Image
        for (unsigned int img = 0; img < this->systems.size(); ++img)
        {
            // Minus the gradient is the total Force here
            this->systems[img]->hamiltonian->Gradient(*configurations[img], Gradient[img]);
            #ifdef SPIRIT_ENABLE_PINNING
                Vectormath::set_c_a(1, Gradient[img], Gradient[img], this->parameters->pinning->mask_unpinned);
            #endif // SPIRIT_ENABLE_PINNING
            
            // Copy out
            Vectormath::set_c_a(-1, Gradient[img], forces[img]);
        }
    }

    template <Solver solver>
    void Method_LLG<solver>::Calculate_Force_Virtual(const std::vector<std::shared_ptr<vectorfield>> & configurations, const std::vector<vectorfield> & forces, std::vector<vectorfield> & forces_virtual)
    {
        using namespace Utility;

        for (unsigned int i=0; i<configurations.size(); ++i)
        {
            auto& image = *configurations[i];
            auto& force = forces[i];
            auto& force_virtual = forces_virtual[i];
            auto& parameters = *this->systems[i]->llg_parameters;

            //////////
            // time steps
            scalar damping = parameters.damping;
            // dt = time_step [ps] * gyromagnetic ratio / mu_B / (1+damping^2) <- not implemented
            scalar dtg = parameters.dt * Constants::gamma / Constants::mu_B / (1 + damping*damping);
            scalar sqrtdtg = dtg / std::sqrt( parameters.dt );
            // STT
            // - monolayer
            scalar a_j = parameters.stt_magnitude;
            Vector3 s_c_vec = parameters.stt_polarisation_normal;
            // - gradient
            scalar b_j = a_j;    // pre-factor b_j = u*mu_s/gamma (see bachelorthesis Constantin)
            scalar beta = parameters.beta;  // non-adiabatic parameter of correction term
            Vector3 je = s_c_vec;// direction of current
            //////////

            // TODO: why the 0.5 everywhere??
            if (parameters.direct_minimization)
            {
                dtg = parameters.dt * Constants::gamma / Constants::mu_B;
                Vectormath::set_c_cross( dtg, image, force, force_virtual);
            }
            else
            {
                Vectormath::set_c_a( dtg, force, force_virtual);
                Vectormath::add_c_cross( dtg * damping, image, force, force_virtual);

                // STT
                if (a_j > 0)
                {

                    if (parameters.stt_use_gradient)
                    {
                        auto& geometry = *this->systems[0]->geometry;
                        auto& boundary_conditions = this->systems[0]->hamiltonian->boundary_conditions;
                        // Gradient approximation for in-plane currents
                        Vectormath::directional_gradient(image, geometry, boundary_conditions, je, s_c_grad); // s_c_grad = (j_e*grad)*S
                        Vectormath::add_c_a    ( dtg * a_j * ( damping - beta ), s_c_grad, force_virtual); // TODO: a_j durch b_j ersetzen 
                        Vectormath::add_c_cross( dtg * a_j * ( 1 + beta * damping ), s_c_grad, image, force_virtual); // TODO: a_j durch b_j ersetzen 
                        // Gradient in current richtung, daher => *(-1)
                    }
                    else
                    {
                        // Monolayer approximation
                        Vectormath::add_c_a    ( -dtg * a_j * ( damping - beta ), s_c_vec, force_virtual);
                        Vectormath::add_c_cross( -dtg * a_j * ( 1 + beta * damping ), s_c_vec, image, force_virtual);
                    }
                }

                // Temperature
                if (parameters.temperature > 0)
                {
                    scalar epsilon = parameters.temperature * Utility::Constants::k_B;//std::sqrt(2.0*parameters.damping / (1.0 + std::pow(parameters.damping, 2)) * parameters.temperature * Utility::Constants::k_B);
                    Vectormath::get_random_vectorfield_unitsphere(parameters.prng, this->xi);
                    Vectormath::add_c_a    ( sqrtdtg * epsilon, this->xi, force_virtual);
                    Vectormath::add_c_cross( sqrtdtg * damping * epsilon, image, this->xi, force_virtual);
                }
            }
            // Apply Pinning
            #ifdef SPIRIT_ENABLE_PINNING
                Vectormath::set_c_a(1, force_virtual, force_virtual, parameters.pinning->mask_unpinned);
            #endif // SPIRIT_ENABLE_PINNING
        }
    }


    template <Solver solver>
    bool Method_LLG<solver>::Converged()
    {
        // Check if all images converged
        return std::all_of(this->force_converged.begin(),
                            this->force_converged.end(),
                            [](bool b) { return b; });
    }

    template <Solver solver>
    void Method_LLG<solver>::Hook_Pre_Iteration()
    {

    }

    template <Solver solver>
    void Method_LLG<solver>::Hook_Post_Iteration()
    {
        // --- Convergence Parameter Update
        // Loop over images to calculate the maximum force components
        for (unsigned int img = 0; img < this->systems.size(); ++img)
        {
            this->force_converged[img] = false;
            auto fmax = this->Force_on_Image_MaxAbsComponent(*(this->systems[img]->spins), this->forces_virtual[img]);
            if (fmax > 0) this->force_max_abs_component = fmax;
            else this->force_max_abs_component = 0;
            if (fmax < this->systems[img]->llg_parameters->force_convergence) this->force_converged[img] = true;
        }

        // --- Image Data Update
        // Update the system's Energy
        // ToDo: copy instead of recalculating
        this->systems[0]->UpdateEnergy();

        // ToDo: How to update eff_field without numerical overhead?
        // systems[0]->effective_field = Gradient[0];
        // Vectormath::scale(systems[0]->effective_field, -1);
        Manifoldmath::project_tangential(this->forces[0], *this->systems[0]->spins);
        Vectormath::set_c_a(1, this->forces[0], this->systems[0]->effective_field);
        // systems[0]->UpdateEffectiveField();

        // TODO: In order to update Rx with the neighbouring images etc., we need the state -> how to do this?

        // --- Renormalize Spins?
        // TODO: figure out specialization of members (Method_LLG should hold Parameters_Method_LLG)
        // if (this->parameters->renorm_sd) {
        //     try {
        //         //Vectormath::Normalize(3, s->nos, s->spins);
        //     }
        //     catch (Exception ex)
        // 	{
        //         if (ex == Exception::Division_by_zero)
        // 		{
        // 			Log(Utility::Log_Level::Warning, Utility::Log_Sender::LLG, "During Iteration Spin = (0,0,0) was detected. Using Random Spin Array");
        //             //Utility::Configurations::Random(s, false);
        //         }
        //         else { throw(ex); }
        //     }

        // }//endif renorm_sd
    }

    template <Solver solver>
    void Method_LLG<solver>::Finalize()
    {
        this->systems[0]->iteration_allowed = false;
    }


    template <Solver solver>
    void Method_LLG<solver>::Save_Current(std::string starttime, int iteration, bool initial, bool final)
    {
        // History save
        this->history["max_torque_component"].push_back(this->force_max_abs_component);
        this->systems[0]->UpdateEnergy();
        this->history["E"].push_back(this->systems[0]->E);
        auto mag = Engine::Vectormath::Magnetization(*this->systems[0]->spins);
        this->history["M_z"].push_back(mag[2]);

        // File save
        if (this->parameters->output_any)
        {
            // Convert indices to formatted strings
            auto s_img  = fmt::format("{:0>2}", this->idx_image);
            int base = (int)log10(this->parameters->n_iterations);
            std::string s_iter = fmt::format("{:0>"+fmt::format("{}",base)+"}", iteration);

            std::string preSpinsFile;
            std::string preEnergyFile;
            if (this->systems[0]->llg_parameters->output_tag_time)
            {
                preSpinsFile = this->parameters->output_folder + "/" + starttime + "_Image-" + s_img + "_Spins";
                preEnergyFile = this->parameters->output_folder + "/" + starttime + "_Image-" + s_img + "_Energy";
            }
            else
            {
                preSpinsFile = this->parameters->output_folder + "/Image-" + s_img + "_Spins";
                preEnergyFile = this->parameters->output_folder + "/_Image-" + s_img + "_Energy";
            }

            // Function to write or append image and energy files
            auto writeOutputConfiguration = [this, preSpinsFile, preEnergyFile, iteration](std::string suffix, bool append)
            {
                // File name
                std::string spinsFile = preSpinsFile + suffix + ".txt";

                // Spin Configuration
                IO::Write_Spin_Configuration(this->systems[0], iteration, spinsFile, append);
            };

            auto writeOutputEnergy = [this, preSpinsFile, preEnergyFile, iteration](std::string suffix, bool append)
            {
                bool normalize = this->systems[0]->llg_parameters->output_energy_divide_by_nspins;

                // File name
                std::string energyFile = preEnergyFile + suffix + ".txt";
                std::string energyFilePerSpin = preEnergyFile + "-perSpin" + suffix + ".txt";

                // Energy
                if (append)
                {
                    // Check if Energy File exists and write Header if it doesn't
                    std::ifstream f(energyFile);
                    if (!f.good()) IO::Write_Energy_Header(*this->systems[0], energyFile);
                    // Append Energy to File
                    IO::Append_System_Energy(*this->systems[0], iteration, energyFile, normalize);
                }
                else
                {
                    IO::Write_Energy_Header(*this->systems[0], energyFile);
                    IO::Append_System_Energy(*this->systems[0], iteration, energyFile, normalize);
                    if (this->systems[0]->llg_parameters->output_energy_spin_resolved)
                    {
                        IO::Write_System_Energy_per_Spin(*this->systems[0], energyFilePerSpin, normalize);
                    }
                }
            };
            
            // Initial image before simulation
            if (initial && this->parameters->output_initial)
            {
                writeOutputConfiguration("-initial", false);
                writeOutputEnergy("-initial", false);
            }
            // Final image after simulation
            else if (final && this->parameters->output_final)
            {
                writeOutputConfiguration("-final", false);
                writeOutputEnergy("-final", false);
            }
            
            // Single file output
            if (this->systems[0]->llg_parameters->output_configuration_step)
            {
                writeOutputConfiguration("_" + s_iter, false);
            }
            if (this->systems[0]->llg_parameters->output_energy_step)
            {
                writeOutputEnergy("_" + s_iter, false);
            }

            // Archive file output (appending)
            if (this->systems[0]->llg_parameters->output_configuration_archive)
            {
                writeOutputConfiguration("-archive", true);
            }
            if (this->systems[0]->llg_parameters->output_energy_archive)
            {
                writeOutputEnergy("-archive", true);
            }

            // Save Log
            Log.Append_to_File();
        }
    }

    // Method name as string
    template <Solver solver>
    std::string Method_LLG<solver>::Name() { return "LLG"; }


    // Template instantiations
    template class Method_LLG<Solver::SIB>;
    template class Method_LLG<Solver::Heun>;
    template class Method_LLG<Solver::Depondt>;
    template class Method_LLG<Solver::NCG>;
    template class Method_LLG<Solver::VP>;
}