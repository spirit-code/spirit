#include <Spirit_Defines.h>
#include <engine/Method_LLG.hpp>
#include <engine/Vectormath.hpp>
#include <data/Spin_System.hpp>
#include <data/Spin_System_Chain.hpp>
#include <utility/IO.hpp>
#include <utility/Logging.hpp>

#include <iostream>
#include <ctime>
#include <math.h>

using namespace Utility;

namespace Engine
{
	template <Solver solver>
    Method_LLG<solver>::Method_LLG(std::shared_ptr<Data::Spin_System> system, int idx_img, int idx_chain) :
		Method_Template<solver>(system->llg_parameters, idx_img, idx_chain)
	{
		// Currently we only support a single image being iterated at once:
		this->systems = std::vector<std::shared_ptr<Data::Spin_System>>(1, system);
		this->SenderName = Utility::Log_Sender::LLG;

		// We assume it is not converged before the first iteration
		this->force_converged = std::vector<bool>(this->systems.size(), false);
		this->force_maxAbsComponent = system->llg_parameters->force_convergence + 1.0;

		// Forces
		this->Gradient = std::vector<vectorfield>(this->systems.size(), vectorfield(this->systems[0]->spins->size()));	// [noi][3nos]

		// History
        this->history = std::map<std::string, std::vector<scalar>>{
			{"max_torque_component", {this->force_maxAbsComponent}},
			{"E", {this->force_maxAbsComponent}},
			{"M_z", {this->force_maxAbsComponent}} };
	}


	template <Solver solver>
	void Method_LLG<solver>::Calculate_Force(std::vector<std::shared_ptr<vectorfield>> configurations, std::vector<vectorfield> & forces)
	{
		// int nos = configurations[0]->size() / 3;
		// this->Force_Converged = std::vector<bool>(configurations.size(), false);
		//this->force_maxAbsComponent = 0;

		// Loop over images to calculate the total force on each Image
		for (unsigned int img = 0; img < this->systems.size(); ++img)
		{
			// Minus the gradient is the total Force here
			this->systems[img]->hamiltonian->Gradient(*configurations[img], Gradient[img]);
			#ifdef SPIRIT_ENABLE_PINNING
				Vectormath::set_c_a(1, Gradient[img], Gradient[img], parameters->pinning->mask_unpinned);
			#endif // SPIRIT_ENABLE_PINNING
			// Vectormath::scale(Gradient[img], -1);
			// Copy out
			Vectormath::set_c_a(-1, Gradient[img], forces[img]);
			// forces[img] = Gradient[img];
		}
	}


	template <Solver solver>
	bool Method_LLG<solver>::Force_Converged()
	{
		for (unsigned int img = 0; img < this->systems.size(); ++img)
		{
			if (this->systems[img]->llg_parameters->temperature > 0 || this->systems[img]->llg_parameters->stt_magnitude > 0)
				return false;
		}
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
			auto fmax = this->Force_on_Image_MaxAbsComponent(*(this->systems[img]->spins), Gradient[img]);
			if (fmax > 0) this->force_maxAbsComponent = fmax;
			else this->force_maxAbsComponent = 0;
			if (fmax < this->systems[img]->llg_parameters->force_convergence) this->force_converged[img] = true;
		}

		// --- Image Data Update
		// Update the system's Energy
		// ToDo: copy instead of recalculating
		this->systems[0]->UpdateEnergy();

		// ToDo: How to update eff_field without numerical overhead?
		// systems[0]->effective_field = Gradient[0];
		// Vectormath::scale(systems[0]->effective_field, -1);
		Vectormath::set_c_a(-1, Gradient[0], this->systems[0]->effective_field);
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
        this->history["max_torque_component"].push_back(this->force_maxAbsComponent);
		this->systems[0]->UpdateEnergy();
        this->history["E"].push_back(this->systems[0]->E);
    	auto mag = Engine::Vectormath::Magnetization(*this->systems[0]->spins);
        this->history["M_z"].push_back(mag[2]);

		// File save
		if (this->parameters->output_any)
		{
			// Convert indices to formatted strings
			auto s_img = IO::int_to_formatted_string(this->idx_image, 2);
			auto s_iter = IO::int_to_formatted_string(iteration, (int)log10(this->parameters->n_iterations));

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
				Utility::IO::Write_Spin_Configuration(this->systems[0], iteration, spinsFile, append);
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
					if (!f.good()) Utility::IO::Write_Energy_Header(*this->systems[0], energyFile);
					// Append Energy to File
					Utility::IO::Append_System_Energy(*this->systems[0], iteration, energyFile, normalize);
				}
				else
				{
					Utility::IO::Write_Energy_Header(*this->systems[0], energyFile);
					Utility::IO::Append_System_Energy(*this->systems[0], iteration, energyFile, normalize);
					if (this->systems[0]->llg_parameters->output_energy_spin_resolved)
					{
						Utility::IO::Write_System_Energy_per_Spin(*this->systems[0], energyFilePerSpin, normalize);
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

	// Optimizer name as string
	template <Solver solver>
    std::string Method_LLG<solver>::Name() { return "LLG"; }


	// Template instantiations
	template class Method_LLG<Solver::SIB>;
	template class Method_LLG<Solver::Heun>;
	template class Method_LLG<Solver::Depondt>;
	template class Method_LLG<Solver::NCG>;
	template class Method_LLG<Solver::VP>;
}