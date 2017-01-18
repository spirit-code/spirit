#include <engine/Method_GNEB.hpp>
#include <data/Spin_System_Chain.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Manifoldmath.hpp>
#include <utility/Cubic_Hermite_Spline.hpp>
#include <utility/IO.hpp>
#include <utility/Timing.hpp>
#include <utility/Logging.hpp>

#include <iostream>
#include <math.h>

using namespace Utility;

namespace Engine
{
    Method_GNEB::Method_GNEB(std::shared_ptr<Data::Spin_System_Chain> chain, int idx_chain) :
		Method(chain->gneb_parameters, -1, idx_chain), chain(chain)
	{
		this->systems = chain->images;
		this->SenderName = Utility::Log_Sender::GNEB;

		int noi = chain->noi;
		int nos = chain->images[0]->nos;

		this->energies = std::vector<scalar>(noi, 0.0);
		this->Rx = std::vector<scalar>(noi, 0.0);

		// We assume that the chain is not converged before the first iteration
		this->force_maxAbsComponent = this->chain->gneb_parameters->force_convergence + 1.0;

		// Tangents
		this->tangents = std::vector<vectorfield>(noi, vectorfield(nos));	// [noi][nos]
		// Forces
		this->F_total    = std::vector<vectorfield>(noi, vectorfield(nos));	// [noi][nos]
		this->F_gradient = std::vector<vectorfield>(noi, vectorfield(nos));	// [noi][nos]
		this->F_spring   = std::vector<vectorfield>(noi, vectorfield(nos));	// [noi][nos]

		// Calculate Data for the border images, which will not be updated
		this->chain->images[0]->UpdateEffectiveField();// hamiltonian->Effective_Field(image, this->chain->images[0]->effective_field);
		this->chain->images[noi-1]->UpdateEffectiveField();//hamiltonian->Effective_Field(image, this->chain->images[0]->effective_field);
	}

	void Method_GNEB::Calculate_Force(std::vector<std::shared_ptr<vectorfield>> configurations, std::vector<vectorfield> & forces)
	{
		int nos = configurations[0]->size();

		// We assume here that we receive a vector of configurations that corresponds to the vector of systems we gave the optimizer.
		//		The Optimizer shuld respect this, but there is no way to enforce it.
		// Get Energy and Gradient of configurations
		for (int i = 0; i < chain->noi; ++i)
		{
			// Calculate the Energy of the image
			energies[i] = this->chain->images[i]->hamiltonian->Energy(*configurations[i]);
			if (i>0) Rx[i] = Rx[i - 1] + Engine::Manifoldmath::dist_geodesic(*configurations[i], *configurations[i - 1]);
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

			// Copy out
			forces[img] = F_total[img];
		}// end for img=1..noi-1
	}// end Calculate

	bool Method_GNEB::Force_Converged()
	{
		// return this->isConverged;
		if (this->force_maxAbsComponent < this->chain->gneb_parameters->force_convergence) return true;
		return false;
	}

	bool Method_GNEB::Iterations_Allowed()
	{
		return this->chain->iteration_allowed;
	}

	void Method_GNEB::Hook_Pre_Iteration()
	{

	}

	void Method_GNEB::Hook_Post_Iteration()
	{
		// --- Convergence Parameter Update
		this->force_maxAbsComponent = 0;
		for (int img = 1; img < chain->noi - 1; ++img)
		{
			scalar fmax = this->Force_on_Image_MaxAbsComponent(*(systems[img]->spins), F_total[img]);
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

	void Method_GNEB::Finalize()
    {
        this->chain->iteration_allowed=false;
    }


	void Method_GNEB::Save_Current(std::string starttime, int iteration, bool initial, bool final)
	{
		if (this->parameters->save_output_any && ( (initial && this->parameters->save_output_initial) || (final && this->parameters->save_output_final) ) )
		{
			// Get the file suffix
			std::string suffix = "";
			if (final) suffix = "_final";
			else suffix = "";

			// always formatting to 6 digits may be problematic!
			auto s_iter = IO::int_to_formatted_string(iteration, 6);

			// Save current Image Chain
			auto imagesFile = this->chain->gneb_parameters->output_folder + "/" + starttime + "_Images_" + s_iter + suffix + ".txt";
			Utility::IO::Save_SpinChain_Configuration(this->chain, imagesFile);

			if (this->parameters->save_output_energy)
			{
				// Save current Energies with reaction coordinates
				auto energiesFile = this->chain->gneb_parameters->output_folder + "/" + starttime + "_E_Images_" + s_iter + suffix + ".txt";
				Utility::IO::Save_Energies(*this->chain, iteration, energiesFile);

				// Save interpolated Energies
				auto energiesInterpFile = this->chain->gneb_parameters->output_folder + "/" + starttime + "_E_interp_Images_" + s_iter + suffix + ".txt";
				Utility::IO::Save_Energies_Interpolated(*this->chain, energiesInterpFile);
			}

			// Save Log
			Log.Append_to_File();
		}
	}

	// Optimizer name as string
    std::string Method_GNEB::Name() { return "GNEB"; }
}