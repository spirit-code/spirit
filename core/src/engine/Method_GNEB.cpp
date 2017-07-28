#include <Spirit_Defines.h>
#include <engine/Method_GNEB.hpp>
#include <data/Spin_System_Chain.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Manifoldmath.hpp>
#include <utility/Cubic_Hermite_Spline.hpp>
#include <utility/IO.hpp>
#include <utility/Logging.hpp>

#include <iostream>
#include <math.h>

using namespace Utility;

namespace Engine
{
	template <Solver solver>
    Method_GNEB<solver>::Method_GNEB(std::shared_ptr<Data::Spin_System_Chain> chain, int idx_chain) :
		Method_Template<solver>(chain->gneb_parameters, -1, idx_chain), chain(chain)
	{
		this->systems = chain->images;
		this->SenderName = Utility::Log_Sender::GNEB;

		int noi = chain->noi;
		int nos = chain->images[0]->nos;

		this->energies = std::vector<scalar>(noi, 0.0);
		this->Rx = std::vector<scalar>(noi, 0.0);

		// History
        this->history = std::map<std::string, std::vector<scalar>>{
			{"max_torque_component", {this->force_maxAbsComponent}} };

		// We assume that the chain is not converged before the first iteration
		this->force_maxAbsComponent = this->chain->gneb_parameters->force_convergence + 1.0;

		// Tangents
		this->tangents = std::vector<vectorfield>(noi, vectorfield( nos, { 0, 0, 0 } ));	// [noi][nos]
		// Forces
		this->F_total    = std::vector<vectorfield>(noi, vectorfield( nos, { 0, 0, 0 } ));	// [noi][nos]
		this->F_gradient = std::vector<vectorfield>(noi, vectorfield( nos, { 0, 0, 0 } ));	// [noi][nos]
		this->F_spring   = std::vector<vectorfield>(noi, vectorfield( nos, { 0, 0, 0 } ));	// [noi][nos]

		// Calculate Data for the border images, which will not be updated
		this->chain->images[0]->UpdateEffectiveField();// hamiltonian->Effective_Field(image, this->chain->images[0]->effective_field);
		this->chain->images[noi-1]->UpdateEffectiveField();//hamiltonian->Effective_Field(image, this->chain->images[0]->effective_field);
	}

	template <Solver solver>
	void Method_GNEB<solver>::Calculate_Force(std::vector<std::shared_ptr<vectorfield>> configurations, std::vector<vectorfield> & forces)
	{
		int nos = configurations[0]->size();

		// We assume here that we receive a vector of configurations that corresponds to the vector of systems we gave the optimizer.
		//		The Optimizer shuld respect this, but there is no way to enforce it.
		// Get Energy and Gradient of configurations
		for (int i = 0; i < chain->noi; ++i)
		{
			// Calculate the Energy of the image
			energies[i] = this->chain->images[i]->hamiltonian->Energy(*configurations[i]);
			if (i>0)
			{
				Rx[i] = Rx[i - 1] + Engine::Manifoldmath::dist_geodesic(*configurations[i], *configurations[i - 1]);
				if (Rx[i] - Rx[i-1] < 1e-10)
				{
        			Log(Log_Level::Error, Log_Sender::GNEB, std::string("The geodesic distance between two images is zero! Stopping..."), -1, this->idx_chain);
					this->chain->iteration_allowed = false;
					return;
				}
			}
		}

		// Calculate relevant tangent to magnetisation sphere, considering also the energies of images
		Engine::Manifoldmath::Tangents(configurations, energies, tangents);

		// Get the total force on the image chain
		// Loop over images to calculate the total force on each Image
		for (int img = 1; img < chain->noi - 1; ++img)
		{
			auto& image = *configurations[img];
			// The gradient force (unprojected) is simply the effective field
			//this->chain->images[img]->hamiltonian->Effective_Field(image, F_gradient[img]);
			// We do it the following way so that the effective field can be e.g. displayed,
			//		while the gradient force is manipulated (e.g. projected)
			this->chain->images[img]->UpdateEffectiveField();
			//this->chain->images[img]->hamiltonian->Effective_Field(image, this->chain->images[img]->effective_field);
			F_gradient[img] = this->chain->images[img]->effective_field;

			// Calculate Force
			if (chain->image_type[img] == Data::GNEB_Image_Type::Climbing)
			{
				// We reverse the component in tangent direction
				Engine::Manifoldmath::invert_parallel(F_gradient[img], tangents[img]);
				// And Spring Force is zero
				F_total[img] = F_gradient[img];
			}
			else if (chain->image_type[img] == Data::GNEB_Image_Type::Falling)
			{
				// Spring Force is zero
				F_total[img] = F_gradient[img];
			}
			else if (chain->image_type[img] == Data::GNEB_Image_Type::Normal)
			{
				// We project the gradient force orthogonal to the TANGENT
				Engine::Manifoldmath::project_orthogonal(F_gradient[img], tangents[img]);

				// Calculate the spring force
				scalar d = this->chain->gneb_parameters->spring_constant * (Rx[img+1] - 2*Rx[img] + Rx[img-1]);
				Vectormath::set_c_a(d, tangents[img], F_spring[img]);
				// for (int i = 0; i < nos; ++i)
				// {
				// 	F_spring[img][i] = d * tangents[img][i];
				// }

				// Calculate the total force
				Vectormath::set_c_a(1, F_gradient[img], F_total[img]);
				Vectormath::add_c_a(1, F_spring[img], F_total[img]);
				// for (int j = 0; j < nos; ++j)
				// {
				// 	F_total[img][j] = F_gradient[img][j] + F_spring[img][j];
				// }
			}
			else
			{
				Vectormath::fill(F_total[img], { 0,0,0 });
			}
			// Apply pinning mask
			#ifdef SPIRIT_ENABLE_PINNING
				Vectormath::set_c_a(1, F_total[img], F_total[img], parameters->pinning->mask_unpinned);
			#endif // SPIRIT_ENABLE_PINNING

			// Copy out
			forces[img] = F_total[img];
		}// end for img=1..noi-1
	}// end Calculate

	template <Solver solver>
	bool Method_GNEB<solver>::Force_Converged()
	{
		// return this->isConverged;
		if (this->force_maxAbsComponent < this->chain->gneb_parameters->force_convergence) return true;
		return false;
	}

	template <Solver solver>
	bool Method_GNEB<solver>::Iterations_Allowed()
	{
		return this->chain->iteration_allowed;
	}

	template <Solver solver>
	void Method_GNEB<solver>::Hook_Pre_Iteration()
	{

	}

	template <Solver solver>
	void Method_GNEB<solver>::Hook_Post_Iteration()
	{
		// --- Convergence Parameter Update
		this->force_maxAbsComponent = 0;
		for (int img = 1; img < chain->noi - 1; ++img)
		{
			scalar fmax = this->Force_on_Image_MaxAbsComponent(*(this->systems[img]->spins), F_total[img]);
			// TODO: how to handle convergence??
			// if (fmax > this->parameters->force_convergence) this->isConverged = false;
			if (fmax > this->force_maxAbsComponent) this->force_maxAbsComponent = fmax;
		}

		// --- Chain Data Update
		// Calculate the inclinations at the data points
		std::vector<scalar> dE_dRx(chain->noi, 0);
		for (int i = 0; i < chain->noi; ++i)
		{
			// dy/dx
			for (int j = 0; j < chain->images[i]->nos; ++j)
			{
				dE_dRx[i] += this->chain->images[i]->effective_field[j].dot(this->tangents[i][j]);
			}
		}
		// Interpolate data points
		auto interp = Utility::Cubic_Hermite_Spline::Interpolate(this->Rx, this->energies, dE_dRx, chain->gneb_parameters->n_E_interpolations);
		// Update the chain
		//		Rx
		chain->Rx = this->Rx;
		//		E
		for (int img = 1; img < chain->noi; ++img) chain->images[img]->E = this->energies[img];
		//		Rx interpolated
		chain->Rx_interpolated = interp[0];
		//		E interpolated
		chain->E_interpolated  = interp[1];
	}

	template <Solver solver>
	void Method_GNEB<solver>::Finalize()
    {
        this->chain->iteration_allowed=false;
    }


	template <Solver solver>
	void Method_GNEB<solver>::Save_Current(std::string starttime, int iteration, bool initial, bool final)
	{
		// History save
        this->history["max_torque_component"].push_back(this->force_maxAbsComponent);

		// File save
		if (this->parameters->output_any)
		{
			// always formatting to 6 digits may be problematic!
			auto s_iter = IO::int_to_formatted_string(iteration, 6);

			std::string preChainFile;
			std::string preEnergiesFile;
			if (this->systems[0]->llg_parameters->output_tag_time)
			{
				preChainFile = this->parameters->output_folder + "/" + starttime + "_Chain";
				preEnergiesFile = this->parameters->output_folder + "/" + starttime + "_Chain_Energies";
			}
			else
			{
				preChainFile = this->parameters->output_folder + "/Chain";
				preEnergiesFile = this->parameters->output_folder + "/Chain_Energies";
			}

			// Function to write or append image and energy files
			auto writeOutputChain = [this, preChainFile, preEnergiesFile, iteration](std::string suffix)
			{
				// File name
				std::string chainFile = preChainFile + suffix + ".txt";

				// Chain
				Utility::IO::Save_SpinChain_Configuration(this->chain, iteration, chainFile);
			};

			auto writeOutputEnergies = [this, preChainFile, preEnergiesFile, iteration](std::string suffix)
			{
				bool normalize = this->chain->gneb_parameters->output_energies_divide_by_nspins;

				// File name
				std::string energiesFile = preEnergiesFile + suffix + ".txt";
				std::string energiesFileInterpolated = preEnergiesFile + "-interpolated" + suffix + ".txt";
				// std::string energiesFilePerSpin = preEnergiesFile + "PerSpin" + suffix + ".txt";

				// Energies
				Utility::IO::Write_Chain_Energies(*this->chain, iteration, energiesFile, normalize);

				// Interpolated Energies
				if (this->chain->gneb_parameters->output_energies_interpolated)
				{
					Utility::IO::Write_Chain_Energies_Interpolated(*this->chain, energiesFileInterpolated, normalize);
				}
				/*if (this->systems[0]->llg_parameters->output_energy_spin_resolved)
				{
					Utility::IO::Write_System_Energy_per_Spin(*this->systems[0], energiesFilePerSpin, normalize);
				}*/
			};


			// Initial chain before simulation
			if (initial && this->parameters->output_initial)
			{
				writeOutputChain("-initial");
				writeOutputEnergies("-initial");
			}
			// Final chain after simulation
			else if (final && this->parameters->output_final)
			{
				writeOutputChain("-final");
				writeOutputEnergies("-final");
			}

			// Single file output
			if (this->chain->gneb_parameters->output_chain_step)
			{
				writeOutputChain("_" + s_iter);
			}
			if (this->chain->gneb_parameters->output_energies_step)
			{
				writeOutputEnergies("_" + s_iter);
			}

			// Save Log
			Log.Append_to_File();
		}
	}


	template <Solver solver>
	void Method_GNEB<solver>::Lock()
	{
		this->chain->Lock();
	}

	template <Solver solver>
	void Method_GNEB<solver>::Unlock()
	{
		this->chain->Unlock();
	}

	// Optimizer name as string
	template <Solver solver>
    std::string Method_GNEB<solver>::Name() { return "GNEB"; }
}