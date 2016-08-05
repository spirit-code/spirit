
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
		Method(system->llg_parameters, idx_img, idx_chain), system(system)
	{
		this->systems = std::vector<std::shared_ptr<Data::Spin_System>>(1, system);
		// // Method child-class specific instructions
		// // currently only support a single image being iterated:
		// this->force_call = std::shared_ptr<Engine::Force>(new Force_LLG(this->c));
		// this->systems.push_back(c->images[c->idx_active_image]);
	}


	void Method_LLG::Calculate_Force(std::vector<std::shared_ptr<std::vector<double>>> configurations, std::vector<std::vector<double>> & forces)
	{
		int noi = configurations.size();
		int nos = configurations[0]->size() / 3;
		// this->Force_Converged = std::vector<bool>(configurations.size(), false);
		this->force_maxAbsComponent = 0;

		// TODO: override Force convergence stuff
		// Loop over images to calculate the total Effective Field on each Image
		for (int img = 0; img < noi; ++img)
		{
			// The gradient force (unprojected) is simply the effective field
			systems[img]->hamiltonian->Effective_Field(*configurations[img], forces[img]);
			// Check for convergence
			auto fmax = this->Force_on_Image_MaxAbsComponent(*configurations[img], forces[img]);
			// if (fmax < this->parameters->force_convergence) this->Force_Converged[img] = true;
			if (fmax > this->force_maxAbsComponent) this->force_maxAbsComponent = fmax;
		}
	}


	bool Method_LLG::Force_Converged()
	{
		return false;
		/*return std::all_of(this->isConverged.begin(),
							this->isConverged.end(),
							[](bool b) { return b; });*/
	}

	void Method_LLG::Hook_Pre_Step()
    {

	}

    void Method_LLG::Hook_Post_Step()
    {
		// this->optimizer->Step();

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

	
	void Method_LLG::Save_Step(int iteration, bool final)
	{
		// TODO: how to do this??
		//		how to get the start time
		// 		how to get the spin configurations
		// Convert image to a formatted string
		auto s_img = IO::int_to_formatted_string(this->idx_image, 2);
		auto s_iter = IO::int_to_formatted_string(iteration, 6);

		// Append Spin configuration to File
		// auto spinsFile = this->parameters->output_folder + "/" + this->starttime + "_" + "Spins_" + s_img + suffix + ".txt";
		// Utility::IO::Append_Spin_Configuration(this->c->images[image], iteration, spinsFile);

		// Save Spin configuration to new File
		// auto spinsIterFile = this->parameters->output_folder + "/" + this->starttime + "_" + "Spins_" + s_img + "_" + s_iter + ".txt";
		// Utility::IO::Append_Spin_Configuration(this->c->images[image], iteration, spinsIterFile);

		// Check if Energy File exists and write Header if it doesn't
		// auto energyFile = this->parameters->output_folder + "/" + this->starttime + "_Energy_" + s_img + suffix + ".txt";
		// std::ifstream f(energyFile);
		// if (!f.good()) Utility::IO::Write_Energy_Header(energyFile);
		
		// Append Energy to File
		// Utility::IO::Append_Energy(*(this->c->images[image]), iteration, energyFile);

		// Save Log
		Log.Append_to_File();
	}

	// Optimizer name as string
    std::string Method_LLG::Name() { return "LLG"; }
}