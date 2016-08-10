
#include "Method_LLG.h"

#include "Optimizer_Heun.h"

#include "Spin_System.h"
#include "Spin_System_Chain.h"
#include "Vectormath.h"
#include "IO.h"
#include "Configurations.h"
#include "Timing.h"
#include "Exception.h"

#include <iostream>
#include <ctime>
#include <math.h>

#include"Logging.h"

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

		// Forces
		this->F_total = std::vector<std::vector<double>>(systems.size(), std::vector<double>(systems[0]->spins->size()));	// [noi][3nos]
	}


	void Method_LLG::Calculate_Force(std::vector<std::shared_ptr<std::vector<double>>> configurations, std::vector<std::vector<double>> & forces)
	{
		// int nos = configurations[0]->size() / 3;
		// this->Force_Converged = std::vector<bool>(configurations.size(), false);
		this->force_maxAbsComponent = 0;

		// TODO: override Force convergence stuff
		// Loop over images to calculate the total Effective Field on each Image
		for (unsigned int img = 0; img < systems.size(); ++img)
		{
			// The effective field is the total Force here
			systems[img]->hamiltonian->Effective_Field(*configurations[img], F_total[img]);
			// Copy out
			forces[img] = F_total[img];
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
		this->force_maxAbsComponent = 0;
		// Loop over images to calculate the total Effective Field on each Image
		for (unsigned int img = 0; img < systems.size(); ++img)
		{
			this->force_converged[img] = false;
			auto fmax = this->Force_on_Image_MaxAbsComponent(*(systems[img]->spins), F_total[img]);
			if (fmax > this->force_maxAbsComponent) this->force_maxAbsComponent = fmax;
			if (fmax < this->systems[img]->llg_parameters->force_convergence) this->force_converged[img] = true;
		}

		// --- Image Data Update
		// Update the system's Energy
		systems[0]->UpdateEnergy();

		// --- Renormalize Spins?
		// TODO: figure out specialization of members (Method_LLG should hold Parameters_LLG)
        // if (this->parameters->renorm_sd) {
        //     try {
        //         //Vectormath::Normalize(3, s->nos, s->spins);
        //     }
        //     catch (Exception ex)
		// 	{
        //         if (ex == Exception::Division_by_zero)
		// 		{
		// 			Log.Send(Utility::Log_Level::WARNING, Utility::Log_Sender::LLG, "During Iteration Spin = (0,0,0) was detected. Using Random Spin Array");
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

	
	void Method_LLG::Save_Current(std::string starttime, int iteration, bool final)
	{
		// Convert indices to formatted strings
		auto s_img = IO::int_to_formatted_string(this->idx_image, 2);
		auto s_iter = IO::int_to_formatted_string(iteration, 6);
		
		std::string suffix = "";
		
		if (final)
		{
			auto s_fix = "_" + IO::int_to_formatted_string(iteration, (int)log10(this->parameters->n_iterations)) + "_final";
			suffix = s_fix;
		}
		else suffix = "_archive";

		// Append Spin configuration to Spin_Archieve_File
		auto spinsFile = this->parameters->output_folder + "/" + starttime + "_" + "Spins_" + s_img + suffix + ".txt";
		Utility::IO::Append_Spin_Configuration(this->systems[0], iteration, spinsFile);

		if (this->systems[0]->llg_parameters->save_single_configurations) {
			// Save Spin configuration to new "spins" File
			auto spinsIterFile = this->parameters->output_folder + "/" + starttime + "_" + "Spins_" + s_img + "_" + s_iter + ".txt";
			Utility::IO::Append_Spin_Configuration(this->systems[0], iteration, spinsIterFile);
		}
		
		// Check if Energy File exists and write Header if it doesn't
		auto energyFile = this->parameters->output_folder + "/" + starttime + "_Energy_" + s_img + suffix + ".txt";
		std::ifstream f(energyFile);
		if (!f.good()) Utility::IO::Write_Energy_Header(energyFile);
		// Append Energy to File
		Utility::IO::Append_Energy(*this->systems[0], iteration, energyFile);

		// Save Log
		Log.Append_to_File();
	}

	// Optimizer name as string
    std::string Method_LLG::Name() { return "LLG"; }
}