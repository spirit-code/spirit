#include <engine/Method_GNEB.hpp>
#include <engine/Vectormath.hpp>
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
		this->tangents = std::vector<std::vector<Vector3>>(noi, std::vector<Vector3>(nos));	// [noi][nos]
		// Forces
		this->F_total    = std::vector<std::vector<Vector3>>(noi, std::vector<Vector3>(nos));	// [noi][nos]
		this->F_gradient = std::vector<std::vector<Vector3>>(noi, std::vector<Vector3>(nos));	// [noi][nos]
		this->F_spring   = std::vector<std::vector<Vector3>>(noi, std::vector<Vector3>(nos));	// [noi][nos]
	}

	void Method_GNEB::Calculate_Force(std::vector<std::shared_ptr<std::vector<Vector3>>> configurations, std::vector<std::vector<Vector3>> & forces)
	{
		int nos = configurations[0]->size();

		// We assume here that we receive a vector of configurations that corresponds to the vector of systems we gave the optimizer.
		//		The Optimizer shuld respect this, but there is no way to enforce it.
		// Get Energy and Effective Field of configurations
		for (int i = 0; i < chain->noi; ++i)
		{
			// Calculate the Energy of the image
			energies[i] = this->chain->images[i]->hamiltonian->Energy(*configurations[i]);
			if (i>0) Rx[i] = Rx[i - 1] + Engine::Vectormath::dist_geodesic(*configurations[i], *configurations[i - 1]);
		}

		// Calculate relevant tangent to magnetisation sphere, considering also the energies of images
		Engine::Vectormath::Tangents(configurations, energies, tangents);

		// Get the total force on the image chain
		// Loop over images to calculate the total Effective Field on each Image
		for (int img = 1; img < chain->noi - 1; ++img)
		{
			auto& image = *configurations[img];
			// The gradient force (unprojected) is simply the effective field
			//this->chain->images[img]->hamiltonian->Effective_Field(image, F_gradient[img]);
			// We do it the following way so that the effective field can be e.g. displayed,
			//		while the gradient force is manipulated (e.g. projected)
			this->chain->images[img]->hamiltonian->Effective_Field(image, this->chain->images[img]->effective_field);
			F_gradient[img] = this->chain->images[img]->effective_field;

			// Calculate Force
			if (chain->climbing_image[img])
			{
				// We reverse the component in tangent direction
				Engine::Vectormath::Project_Reverse(F_gradient[img], tangents[img]);
				// And Spring Force is zero
				F_total[img] = F_gradient[img];
			}
			else if (chain->falling_image[img])
			{
				// We project the gradient force orthogonal to the tangent
				// If anything, project orthogonal to the spins... idiot! But Heun already does that.
				//Utility::Manifoldmath::Project_Orthogonal(F_gradient[img], this->c->tangents[img]);
				// Spring Force is zero
				F_total[img] = F_gradient[img];
			}
			else
			{
				// We project the gradient force orthogonal to the SPIN
				//Utility::Manifoldmath::Project_Orthogonal(F_gradient[img], this->c->tangents[img]);
				// Get the scalar product of the vectors
				// scalar v1v2 = 0.0;
				// int dim;
				// // Take out component in direction of v2
				// for (int i = 0; i < nos; ++i)
				// {
				// 	v1v2 = 0.0;
				// 	for (dim = 0; dim < 3; ++dim)
				// 	{
				// 		v1v2 += F_gradient[img][i+dim*nos] * image[i+dim*nos];
				// 	}
				// 	for (dim = 0; dim < 3; ++dim)
				// 	{
				// 		F_gradient[img][i + dim*nos] = F_gradient[img][i + dim*nos] - v1v2 * image[i + dim*nos];
				// 	}
				// }
			

				// We project the gradient force orthogonal to the TANGENT
				//Utility::Manifoldmath::Project_Orthogonal(F_gradient[img], this->c->tangents[img]);
				// Get the scalar product of the vectors
				scalar v1v2 = 0.0;
				for (int i = 0; i < nos; ++i)
				{
					v1v2 += F_gradient[img][i].dot(tangents[img][i]);
				}
				// Take out component in direction of v2
				for (int i = 0; i < nos; ++i)
				{
					F_gradient[img][i] -= v1v2 * tangents[img][i];
				}


				// Calculate the spring force
				//spring_forces(:, : ) = spring_constant *(dist_geodesic(NOS, IMAGES_LAST(idx_img + 1, :, : ), IMAGES(idx_img, :, : )) - dist_geodesic(NOS, IMAGES(idx_img, :, : ), IMAGES_LAST(idx_img - 1, :, : )))* tangents(:, : );
				scalar d = this->chain->gneb_parameters->spring_constant * (Rx[img+1] - 2*Rx[img] + Rx[img-1]);
				for (int i = 0; i < nos; ++i)
				{
					F_spring[img][i] = d * tangents[img][i];
				}

				// Calculate the total force
				for (int j = 0; j < nos; ++j)
				{
					F_total[img][j] = F_gradient[img][j] + F_spring[img][j];
				}
			}// end if climbing

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

		// Get the file suffix
		std::string suffix = "";
		if (final) suffix = "_final";
		else suffix = "";

		// always formatting to 6 digits may be problematic!
		auto s_iter = IO::int_to_formatted_string(iteration, 6);

		// Save current Image Chain
		auto imagesFile = this->chain->gneb_parameters->output_folder + "/" + starttime + "_Images_" + s_iter + suffix + ".txt";
		Utility::IO::Save_SpinChain_Configuration(this->chain, imagesFile);

		// Save current Energies with reaction coordinates
		auto energiesFile = this->chain->gneb_parameters->output_folder + "/" + starttime + "_E_Images_" + s_iter + suffix + ".txt";
		//		Check if Energy File exists and write Header if it doesn't
		std::ifstream f(energiesFile);
		if (!f.good()) Utility::IO::Write_Energy_Header(energiesFile);
		//		Save
		Utility::IO::Save_Energies(*this->chain, iteration, energiesFile);

		// Save interpolated Energies
		auto energiesInterpFile = this->chain->gneb_parameters->output_folder + "/" + starttime + "_E_interp_Images_" + s_iter + suffix + ".txt";
		Utility::IO::Save_Energies_Interpolated(*this->chain, energiesInterpFile);

		// Save Log
		Log.Append_to_File();
	}

	// Optimizer name as string
    std::string Method_GNEB::Name() { return "GNEB"; }
}