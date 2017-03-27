#include <engine/Method_LLG.hpp>
#include <engine/Optimizer_Heun.hpp>
#include <engine/Vectormath.hpp>
#include <data/Spin_System.hpp>
#include <data/Spin_System_Chain.hpp>
#include <utility/IO.hpp>
#include <utility/Configurations.hpp>
#include <utility/Timing.hpp>
#include <utility/Exception.hpp>
#include <utility/Logging.hpp>

#include <iostream>
#include <ctime>
#include <math.h>

using namespace Utility;

namespace Engine
{
    Method_LLG::Method_LLG(std::shared_ptr<Data::Spin_System> system, int idx_img, int idx_chain) :
		Method(system->llg_parameters, idx_img, idx_chain)
	{
		// Currently we only support a single image being iterated at once:
		this->systems = std::vector<std::shared_ptr<Data::Spin_System>>(1, system);
		this->SenderName = Utility::Log_Sender::LLG;

		// We assume it is not converged before the first iteration
		this->force_converged = std::vector<bool>(systems.size(), false);
		this->force_maxAbsComponent = system->llg_parameters->force_convergence + 1.0;

		// Forces
		this->Gradient = std::vector<vectorfield>(systems.size(), vectorfield(systems[0]->spins->size()));	// [noi][3nos]
		
		// History
        this->history = std::map<std::string, std::vector<scalar>>{
			{"max_torque_component", {this->force_maxAbsComponent}},
			{"E", {this->force_maxAbsComponent}},
			{"M_z", {this->force_maxAbsComponent}} };
	}


	void Method_LLG::Calculate_Force(std::vector<std::shared_ptr<vectorfield>> configurations, std::vector<vectorfield> & forces)
	{
		// int nos = configurations[0]->size() / 3;
		// this->Force_Converged = std::vector<bool>(configurations.size(), false);
		//this->force_maxAbsComponent = 0;

		// Loop over images to calculate the total force on each Image
		for (unsigned int img = 0; img < systems.size(); ++img)
		{
			// Minus the gradient is the total Force here
			systems[img]->hamiltonian->Gradient(*configurations[img], Gradient[img]);
			// Vectormath::scale(Gradient[img], -1);
			// Copy out
			Vectormath::set_c_a(-1, Gradient[img], forces[img]);
			// forces[img] = Gradient[img];
		}
	}


	bool Method_LLG::Force_Converged()
	{
		// Check if all images converged
		return std::all_of(this->force_converged.begin(),
							this->force_converged.end(),
							[](bool b) { return b; });
	}

	void Method_LLG::Hook_Pre_Iteration()
    {

	}

    void Method_LLG::Hook_Post_Iteration()
    {
		// --- Convergence Parameter Update
		// Loop over images to calculate the maximum force components
		for (unsigned int img = 0; img < systems.size(); ++img)
		{
			this->force_converged[img] = false;
			auto fmax = this->Force_on_Image_MaxAbsComponent(*(systems[img]->spins), Gradient[img]);
			if (fmax > 0) this->force_maxAbsComponent = fmax;
			else this->force_maxAbsComponent = 0;
			if (fmax < this->systems[img]->llg_parameters->force_convergence) this->force_converged[img] = true;
		}

		// --- Image Data Update
		// Update the system's Energy
		// ToDo: copy instead of recalculating
		systems[0]->UpdateEnergy();

		// ToDo: How to update eff_field without numerical overhead?
		// systems[0]->effective_field = Gradient[0];
		// Vectormath::scale(systems[0]->effective_field, -1);
		Vectormath::set_c_a(-1, Gradient[0], systems[0]->effective_field);
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

	void Method_LLG::Finalize()
    {
		this->systems[0]->iteration_allowed = false;
    }

	
	void Method_LLG::Save_Current(std::string starttime, int iteration, bool initial, bool final)
	{
		// History save
        this->history["max_torque_component"].push_back(this->force_maxAbsComponent);
		systems[0]->UpdateEnergy();
        this->history["E"].push_back(systems[0]->E);
    	auto mag = Engine::Vectormath::Magnetization(*systems[0]->spins);
        this->history["M_z"].push_back(mag[2]);

		// File save
		if (this->parameters->save_output_any)
		{
			// Convert indices to formatted strings
			auto s_img = IO::int_to_formatted_string(this->idx_image, 2);
			auto s_iter = IO::int_to_formatted_string(iteration, (int)log10(this->parameters->n_iterations));

			std::string preSpinsFile  = this->parameters->output_folder + "/" + starttime + "_" + "Spins_" + s_img;
			std::string preEnergyFile = this->parameters->output_folder + "/" + starttime + "_" + "Energy_" + s_img;

			// Function to write or append image and energy files
			auto writeoutput = [this, preSpinsFile, preEnergyFile, iteration](std::string suffix, bool append)
			{
				// File names
				std::string spinsFile = preSpinsFile + suffix + ".txt";
				std::string energyFile = preEnergyFile + suffix + ".txt";

				// Spin Configuration
				if (append)
				{
					Utility::IO::Append_Spin_Configuration(this->systems[0], iteration, spinsFile);
				}
				else
				{
					Utility::IO::Append_Spin_Configuration(this->systems[0], iteration, spinsFile);
				}
				// Energy
				if (this->parameters->save_output_energy)
				{
					if (append)
					{
						// Check if Energy File exists and write Header if it doesn't
						std::ifstream f(energyFile);
						if (!f.good()) Utility::IO::Write_Energy_Header(*this->systems[0], energyFile);
						// Append Energy to File
						Utility::IO::Append_Energy(*this->systems[0], iteration, energyFile);
					}
					else
					{
						Utility::IO::Write_Energy_Header(*this->systems[0], energyFile);
						Utility::IO::Append_Energy(*this->systems[0], iteration, energyFile);
					}
				}
			};
			
			// Initial image before simulation
			if (initial && this->parameters->save_output_initial)
			{
				writeoutput("_" + s_iter + "_initial", false);
			}
			// Final image after simulation
			else if (final && this->parameters->save_output_final)
			{
				writeoutput("_" + s_iter + "_final", false);
			}
			
			// Single image file output
			if (this->systems[0]->llg_parameters->save_output_single)
			{
				writeoutput("_" + s_iter, false);
			}

			// Archive file (appending)
			if (this->systems[0]->llg_parameters->save_output_archive)
			{
				writeoutput("_archive", true);
			}

			// Save Log
			Log.Append_to_File();
		}
	}

	// Optimizer name as string
    std::string Method_LLG::Name() { return "LLG"; }
}