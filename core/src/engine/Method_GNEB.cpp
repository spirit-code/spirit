#include <Spirit_Defines.h>
#include <engine/Method_GNEB.hpp>
#include <data/Spin_System_Chain.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Manifoldmath.hpp>
#include <io/IO.hpp>
#include <utility/Cubic_Hermite_Spline.hpp>
#include <utility/Logging.hpp>

#include <iostream>
#include <math.h>

#include <fmt/format.h>

using namespace Utility;

namespace Engine
{
	template <Solver solver>
    Method_GNEB<solver>::Method_GNEB(std::shared_ptr<Data::Spin_System_Chain> chain, int idx_chain) :
		Method_Solver<solver>(chain->gneb_parameters, -1, idx_chain), chain(chain)
	{
		this->systems = chain->images;
		this->SenderName = Utility::Log_Sender::GNEB;

		this->noi = chain->noi;
		this->nos = chain->images[0]->nos;

		this->energies = std::vector<scalar>(this->noi, 0);
		this->Rx = std::vector<scalar>(this->noi, 0);

		// Forces
		this->forces     = std::vector<vectorfield>(this->noi, vectorfield( this->nos, { 0, 0, 0 } ));
		// this->Gradient = std::vector<vectorfield>(this->noi, vectorfield(this->nos));
		this->F_total    = std::vector<vectorfield>(this->noi, vectorfield( this->nos, { 0, 0, 0 } ));	// [noi][nos]
		this->F_gradient = std::vector<vectorfield>(this->noi, vectorfield( this->nos, { 0, 0, 0 } ));	// [noi][nos]
		this->F_spring   = std::vector<vectorfield>(this->noi, vectorfield( this->nos, { 0, 0, 0 } ));	// [noi][nos]
		this->xi = vectorfield(this->nos, {0,0,0});

		// Tangents
		this->tangents = std::vector<vectorfield>(this->noi, vectorfield( this->nos, { 0, 0, 0 } ));	// [noi][nos]

		// We assume that the chain is not converged before the first iteration
		this->force_max_abs_component = this->chain->gneb_parameters->force_convergence + 1.0;
		this->force_max_abs_component_all = std::vector<scalar>(this->noi, 0);

		// Create shared pointers to the method's systems' spin configurations
		this->configurations = std::vector<std::shared_ptr<vectorfield>>(this->noi);
		for (int i = 0; i<this->noi; ++i) this->configurations[i] = this->systems[i]->spins;

		// History
        this->history = std::map<std::string, std::vector<scalar>>{
			{"max_torque_component", {this->force_max_abs_component}} };

		// Calculate Data for the border images, which will not be updated
		this->chain->images[0]->UpdateEffectiveField();// hamiltonian->Effective_Field(image, this->chain->images[0]->effective_field);
		this->chain->images[this->noi-1]->UpdateEffectiveField();//hamiltonian->Effective_Field(image, this->chain->images[0]->effective_field);
	}

	template <Solver solver>
	std::vector<scalar> Method_GNEB<solver>::getForceMaxAbsComponent_All()
	{
		return this->force_max_abs_component_all;
	}


	template <Solver solver>
	void Method_GNEB<solver>::Calculate_Force(const std::vector<std::shared_ptr<vectorfield>> & configurations, std::vector<vectorfield> & forces)
	{
		int nos = configurations[0]->size();

		// We assume here that we receive a vector of configurations that corresponds to the vector of systems we gave the Solver.
		//		The Solver shuld respect this, but there is no way to enforce it.
		// Get Energy and Gradient of configurations
		for (int img = 0; img < chain->noi; ++img)
		{
			auto& image = *configurations[img];

			// Calculate the Energy of the image
			energies[img] = this->chain->images[img]->hamiltonian->Energy(image);
			if (img > 0)
			{
				Rx[img] = Rx[img-1] + Manifoldmath::dist_geodesic(image, *configurations[img-1]);
				if (Rx[img] - Rx[img-1] < 1e-10)
				{
        			Log(Log_Level::Error, Log_Sender::GNEB, std::string("The geodesic distance between two images is zero! Stopping..."), -1, this->idx_chain);
					this->chain->iteration_allowed = false;
					return;
				}
			}
		}

		// Calculate relevant tangent to magnetisation sphere, considering also the energies of images
		Manifoldmath::Tangents(configurations, energies, tangents);

		// Get the total force on the image chain
		// Loop over images to calculate the total force on each Image
		for (int img = 1; img < chain->noi - 1; ++img)
		{
			auto& image = *configurations[img];
			// We do it the following way so that the effective field can be e.g. displayed,
			//		while the gradient force is manipulated (e.g. projected)
			this->chain->images[img]->UpdateEffectiveField();
			// F_gradient[img] = this->chain->images[img]->effective_field;
			Vectormath::set_c_a(1, this->chain->images[img]->effective_field, F_gradient[img]);
			// // this->chain->images[img]->hamiltonian->Effective_Field(image, this->chain->images[img]->effective_field);

			// The gradient force (unprojected) is simply the effective field
			// this->chain->images[img]->hamiltonian->Gradient(image, F_gradient[img]);
			// Vectormath::scale(F_gradient[img], -1);

			// Project the gradient force into the tangent space of the image
			Manifoldmath::project_tangential(F_gradient[img], image);

			// Calculate Force
			if (chain->image_type[img] == Data::GNEB_Image_Type::Climbing)
			{
				// We reverse the component in tangent direction
				Manifoldmath::invert_parallel(F_gradient[img], tangents[img]);
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
				Manifoldmath::project_orthogonal(F_gradient[img], tangents[img]);

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
				Vectormath::set_c_a(1, F_total[img], F_total[img], this->parameters->pinning->mask_unpinned);
			#endif // SPIRIT_ENABLE_PINNING

			// Copy out
			Vectormath::set_c_a(1, F_total[img], forces[img]);
		}// end for img=1..noi-1
	}// end Calculate


	template <Solver solver>
	void Method_GNEB<solver>::Calculate_Force_Virtual(const std::vector<std::shared_ptr<vectorfield>> & configurations, const std::vector<vectorfield> & forces, std::vector<vectorfield> & forces_virtual)
    {
		using namespace Utility;

		// Calculate the cross product with the spin configuration to get direct minimization
		for (unsigned int i = 1; i < configurations.size()-1; ++i)
		{
			auto& image = *configurations[i];
			auto& force = forces[i];
			auto& force_virtual = forces_virtual[i];
			auto& parameters = *this->systems[i]->llg_parameters;

			// dt = time_step [ps] * gyromagnetic ratio / mu_B / (1+damping^2) <- not implemented
			scalar dtg = parameters.dt * Constants::gamma / Constants::mu_B;
			Vectormath::set_c_cross( dtg, image, force, force_virtual);

			// TODO: add Temperature effects!

			// Apply Pinning
			#ifdef SPIRIT_ENABLE_PINNING
			Vectormath::set_c_a(1, force_virtual, force_virtual, parameters.pinning->mask_unpinned);
			#endif // SPIRIT_ENABLE_PINNING
		}
    }

	template <Solver solver>
	bool Method_GNEB<solver>::Converged()
	{
		// return this->isConverged;
		if (this->force_max_abs_component < this->chain->gneb_parameters->force_convergence) return true;
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
		this->force_max_abs_component = 0;
		std::fill(this->force_max_abs_component_all.begin(), this->force_max_abs_component_all.end(), 0);
		
		
		for (int img = 1; img < chain->noi - 1; ++img)
		{
			scalar fmax = this->Force_on_Image_MaxAbsComponent(*(this->systems[img]->spins), F_total[img]);
			// Set maximum per image
			if (fmax > this->force_max_abs_component_all[img]) this->force_max_abs_component_all[img] = fmax;
			// Set maximum overall
			if (fmax > this->force_max_abs_component) this->force_max_abs_component = fmax;

			// Set the effective fields
			Manifoldmath::project_tangential(this->forces[img], *this->systems[img]->spins);
        	// Vectormath::set_c_a(1, this->forces[img], this->systems[img]->effective_field);
		}

		// --- Chain Data Update
		// Calculate the inclinations at the data points
		std::vector<scalar> dE_dRx(chain->noi, 0);
		for (int i = 0; i < chain->noi; ++i)
		{
			// dy/dx
			dE_dRx[i] = Vectormath::dot(this->chain->images[i]->effective_field, this->tangents[i]);
			// for (int j = 0; j < chain->images[i]->nos; ++j)
			// {
			// 	dE_dRx[i] += this->chain->images[i]->effective_field[j].dot(this->tangents[i][j]);
			// }
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
        this->history["max_torque_component"].push_back(this->force_max_abs_component);

		// File save
		if (this->parameters->output_any)
		{
			// always formatting to 6 digits may be problematic!
			std::string s_iter = fmt::format("{:0>6}", iteration);

			std::string preChainFile;
			std::string preEnergiesFile;
            std::string fileTag;
            
            if (this->parameters->output_file_tag == "<time>")
                fileTag = starttime + "_";
            else if (this->parameters->output_file_tag != "")
                fileTag = this->parameters->output_file_tag + "_";
            else 
                fileTag = "";
            
            preChainFile = this->parameters->output_folder + "/" + fileTag + "Chain";
            preEnergiesFile = this->parameters->output_folder + "/" + fileTag + "Chain_Energies";

			// Function to write or append image and energy files
            auto writeOutputChain = [ this, preChainFile, preEnergiesFile, iteration ]
                                        ( std::string suffix, bool append )
			{
				// File name
				std::string chainFile = preChainFile + suffix + ".txt";

				// Chain
                std::string output_comment = fmt::format( "Iteration: {}", iteration );
                IO::Write_Chain_Spin_Configuration( this->chain, chainFile, 
                                                    IO::VF_FileFormat::SPIRIT_WHITESPACE_SPIN, 
                                                    output_comment, append );
            };

			auto writeOutputEnergies = [this, preChainFile, preEnergiesFile, iteration](std::string suffix)
			{
				bool normalize = this->chain->gneb_parameters->output_energies_divide_by_nspins;

				// File name
				std::string energiesFile = preEnergiesFile + suffix + ".txt";
				std::string energiesFileInterpolated = preEnergiesFile + "-interpolated" + suffix + ".txt";
				// std::string energiesFilePerSpin = preEnergiesFile + "PerSpin" + suffix + ".txt";

				// Energies
				IO::Write_Chain_Energies(*this->chain, iteration, energiesFile, normalize);

				// Interpolated Energies
				if (this->chain->gneb_parameters->output_energies_interpolated)
				{
					IO::Write_Chain_Energies_Interpolated(*this->chain, energiesFileInterpolated, normalize);
				}
				/*if (this->systems[0]->llg_parameters->output_energy_spin_resolved)
				{
					IO::Write_Image_Energy_per_Spin(*this->systems[0], energiesFilePerSpin, normalize);
				}*/
			};


			// Initial chain before simulation
			if (initial && this->parameters->output_initial)
			{
				writeOutputChain( "-initial", false );
				writeOutputEnergies("-initial");
			}
			// Final chain after simulation
			else if (final && this->parameters->output_final)
			{
				writeOutputChain( "-final", false );
				writeOutputEnergies("-final");
			}

			// Single file output
			if (this->chain->gneb_parameters->output_chain_step)
			{
				writeOutputChain("_" + s_iter, false );
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

	// Method name as string
	template <Solver solver>
    std::string Method_GNEB<solver>::Name() { return "GNEB"; }

	// Template instantiations
	template class Method_GNEB<Solver::SIB>;
	template class Method_GNEB<Solver::Heun>;
	template class Method_GNEB<Solver::Depondt>;
	template class Method_GNEB<Solver::NCG>;
	template class Method_GNEB<Solver::VP>;
}