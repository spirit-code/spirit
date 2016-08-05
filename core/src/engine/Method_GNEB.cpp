#include "Method_GNEB.h"

#include "Manifoldmath.h"
#include "Cubic_Hermite_Spline.h"
#include "IO.h"
#include "Timing.h"

#include "Optimizer_Heun.h"
#include "Optimizer_SIB.h"
#include "Vectormath.h"

#include"Logging.h"

#include <iostream>
#include <math.h>

using namespace Utility;

namespace Engine
{
    Method_GNEB::Method_GNEB(std::shared_ptr<Data::Spin_System_Chain> chain, int idx_img, int idx_chain) :
		Method(chain->gneb_parameters, idx_img, idx_chain), chain(chain)
	{
		this->systems = chain->images;
		this->SenderName = Utility::Log_Sender::GNEB;
		// Method child-class specific instructions
		// this->force_call = std::shared_ptr<Engine::Force>(new Force_GNEB(this->c));
		// this->systems = c->images;

		// Configure the Optimizer
		// this->optimizer->Configure(this->systems, this->force_call);
	}

	void Method_GNEB::Calculate_Force(std::vector<std::shared_ptr<std::vector<double>>> configurations, std::vector<std::vector<double>> & forces)
	{
		int noi = configurations.size();
		int nos = configurations[0]->size()/3;
		// this->Force_Converged = false;
		this->force_maxAbsComponent = 0;

		// We assume here that we receive a vector of configurations that corresponds to the vector of systems we gave the optimizer.
		//		The Optimizer shuld respect this, but there is no way to enforce it.
		// Get Energy and Effective Field of configurations
		std::vector<double> energies(configurations.size());
		for (int i = 0; i < noi; ++i)
		{
			// Calculate the Energy of the image (and store it in image)
			// ----- energies[i] = this->c->images[i]->hamiltonian->Energy(configurations[i]);
			
			// NEED A NICER WAY OF DOING THIS ---- THE ENERGY ETC SHOULD NOT BE UPDATED HERE, SINCE THIS MIGHT BE CALLED
			// TO CALCULATE INTERMEDIATE FORCES INSTEAD OF THE FORCES ON THE SPIN SYSTEMS
			
			// ----- this->c->images[i]->E = energies[i];
			
			// Calculate Effective Field
			//c->images[i]->Effective_Field(configurations[i], beff);
			// Calculate relevant tangent to magnetisation sphere, considering also the energies of images
			//Utility::Manifoldmath::Tangent(*c, i, t[i]);
		}

		// Calculate relevant tangent to magnetisation sphere, considering also the energies of images
		// ----- Utility::Manifoldmath::Tangents(configurations, energies, this->c->tangents);

		// Get the total force on the image chain
		//auto force = std::vector<std::vector<double>>(c->noi, std::vector<double>(3 * nos));	// [noi][3nos]

		// Forces
		auto F_gradient = std::vector<std::vector<double>>(noi, std::vector<double>(3 * nos));	// [noi][3nos]
		auto F_spring = std::vector<std::vector<double>>(noi, std::vector<double>(3 * nos));	// [noi][3nos]
		//auto F_total = std::vector<std::vector<double>>(c->noi, std::vector<double>(3 * nos));	// [noi][3nos]
		// Tangents
		//auto t = std::vector<std::vector<double>>(c->noi, std::vector<double>(3 * nos));	// [noi][3nos]

		// Loop over images to calculate the total Effective Field on each Image
		for (int img = 1; img < noi - 1; ++img)
		{
			// TODO: figure out how to handle the images in this case...
			// // The gradient force (unprojected) is simply the effective field
			// this->c->images[img]->hamiltonian->Effective_Field(configurations[img], F_gradient[img]);
			// // NEED A NICER WAY OF DOING THIS:
			// this->c->images[img]->effective_field = F_gradient[img];

			// // Calculate Force
			// if (c->climbing_image[img])
			// {
			// 	// We reverse the component in tangent direction
			// 	Utility::Manifoldmath::Project_Reverse(F_gradient[img], this->c->tangents[img]);
			// 	// And Spring Force is zero
			// 	forces[img] = F_gradient[img];
			// }
			// else if (c->falling_image[img])
			// {
			// 	// We project the gradient force orthogonal to the tangent
			// 	// If anything, project orthogonal to the spins... idiot! But Heun already does that.
			// 	//Utility::Manifoldmath::Project_Orthogonal(F_gradient[img], this->c->tangents[img]);
			// 	// Spring Force is zero
			// 	forces[img] = F_gradient[img];
			// }
			// else
			// {

			// 	// We project the gradient force orthogonal to the SPIN
			// 	//Utility::Manifoldmath::Project_Orthogonal(F_gradient[img], this->c->tangents[img]);
			// 	// Get the scalar product of the vectors
			// 	double v1v2 = 0.0;
			// 	int dim;
			// 	// Take out component in direction of v2
			// 	for (int i = 0; i < nos; ++i)
			// 	{
			// 		v1v2 = 0.0;
			// 		for (dim = 0; dim < 3; ++dim)
			// 		{
			// 			v1v2 += F_gradient[img][i+dim*nos] * configurations[img][i+dim*nos];
			// 		}
			// 		for (dim = 0; dim < 3; ++dim)
			// 		{
			// 			F_gradient[img][i + dim*nos] = F_gradient[img][i + dim*nos] - v1v2 * configurations[img][i + dim*nos];
			// 		}
			// 	}


			// 	// We project the gradient force orthogonal to the TANGENT
			// 	//Utility::Manifoldmath::Project_Orthogonal(F_gradient[img], this->c->tangents[img]);
			// 	// Get the scalar product of the vectors
			// 	v1v2 = 0.0;
			// 	for (int i = 0; i < 3*nos; ++i)
			// 	{
			// 		v1v2 += F_gradient[img][i] * this->c->tangents[img][i];
			// 	}
			// 	// Take out component in direction of v2
			// 	for (int i = 0; i < 3 * nos; ++i)
			// 	{
			// 		F_gradient[img][i] = F_gradient[img][i] - v1v2 * this->c->tangents[img][i];
			// 	}


			// 	// Calculate the spring force
			// 	//spring_forces(:, : ) = spring_constant *(dist_geodesic(NOS, IMAGES_LAST(idx_img + 1, :, : ), IMAGES(idx_img, :, : )) - dist_geodesic(NOS, IMAGES(idx_img, :, : ), IMAGES_LAST(idx_img - 1, :, : )))* tangents(:, : );
			// 	double d1, d2, d;
			// 	d1 = Utility::Manifoldmath::Dist_Geodesic(configurations[img + 1], configurations[img]);
			// 	d2 = Utility::Manifoldmath::Dist_Geodesic(configurations[img], configurations[img - 1]);
			// 	d = this->c->gneb_parameters->spring_constant * (d1 - d2);
			// 	for (unsigned int i = 0; i < F_spring[0].size(); ++i)
			// 	{
			// 		F_spring[img][i] = d * this->c->tangents[img][i];
			// 	}

			// 	// Calculate the total force
			// 	for (int j = 0; j < 3 * nos; ++j)
			// 	{
			// 		forces[img][j] = F_gradient[img][j] + F_spring[img][j];
			// 	}
			// }// end if climbing
		}// end for img=1..noi-1

		// Check for convergence
		for (int img = 1; img < noi - 1; ++img)
		{
			double fmax = this->Force_on_Image_MaxAbsComponent(*configurations[img], forces[img]);
			// TODO: how to handle convergence??
			// if (fmax > this->parameters->force_convergence) this->isConverged = false;
			if (fmax > this->force_maxAbsComponent) this->force_maxAbsComponent = fmax;
		}
	}// end Calculate

	bool Method_GNEB::Force_Converged()
	{
		// return this->isConverged;
		return false;
	}

	void Method_GNEB::Hook_Pre_Step()
	{

	}

	void Method_GNEB::Hook_Post_Step()
	{
		// Update Images' Energies
		for (int i = 0; i < chain->noi; ++i)
		{
			//Engine::Energy::Update(*chain->images[i]);
			//chain->images[i]->E = chain->images[i]->hamiltonian_isotropichain->Energy(chain->images[i]->spins);
			chain->images[i]->UpdateEnergy();
			// if (i > 0) chain->Rx[i] = chain->Rx[i - 1] + Utility::Manifoldmath::Dist_Geodesic(chain->images[i - 1]->spins, state->active_chain->images[i]->spins);
		}

		// Update Chain's reaction coordinates
		for (int i = 1; i < chain->noi; ++i)
		{
			chain->Rx[i] = chain->Rx[i - 1] + Utility::Manifoldmath::Dist_Geodesic(*chain->images[i - 1]->spins, *chain->images[i]->spins);
		}

		// TODO: UPDATE THE TANGENTS??
		// ...

		// Calculate the inclinations at the data points
		std::vector<double> E(chain->noi, 0);
		std::vector<double> dE_dRx(chain->noi, 0);
		for (int i = 0; i < chain->noi; ++i)
		{
			// y array
			E[i] = chain->images[i]->E;
			// dy/dx
			for (int j = 0; j < 3 * chain->images[i]->nos; ++j)
			{
				dE_dRx[i] += chain->images[i]->effective_field[j] * chain->tangents[i][j];
			}
		}

		// Interpolate data points
		auto interp = Utility::Cubic_Hermite_Spline::Interpolate(chain->Rx, E, dE_dRx, chain->gneb_parameters->n_E_interpolations);

		// Assign to chain
		chain->Rx_interpolated = interp[0];
		chain->E_interpolated  = interp[1];
	}


	void Method_GNEB::Save_Step(std::string starttime, int iteration, bool final)
	{

		// Get the file suffix
		std::string suffix = "";
		if (final) suffix = "_final";
		else suffix = "";

		// always formatting to 6 digits may be problematic!
		auto s_iter = IO::int_to_formatted_string(iteration, 6);

		// TODO: how to handle??

		// // Save current Image Chain
		// auto imagesFile = this->c->gneb_parameters->output_folder + "/" + this->starttime + "_Images_" + s_iter + suffix + ".txt";
		// Utility::IO::Save_SpinChain_Configuration(this->c, imagesFile);

		// // Save current Energies with reaction coordinates
		// auto energiesFile = this->c->gneb_parameters->output_folder + "/" + this->starttime + "_E_Images_" + s_iter + suffix + ".txt";
		// //		Check if Energy File exists and write Header if it doesn't
		// std::ifstream f(energiesFile);
		// if (!f.good()) Utility::IO::Write_Energy_Header(energiesFile);
		// //		Save
		// Utility::IO::Save_Energies(*this->c, iteration, energiesFile);

		// // Save interpolated Energies
		// auto energiesInterpFile = this->c->gneb_parameters->output_folder + "/" + this->starttime + "_E_interp_Images_" + s_iter + suffix + ".txt";
		// Utility::IO::Save_Energies_Interpolated(*this->c, energiesInterpFile);

		// Save Log
		Log.Append_to_File();
	}

	// Optimizer name as string
    std::string Method_GNEB::Name() { return "GNEB"; }
}