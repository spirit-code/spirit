#include "Force_GNEB.h"
#include "Manifoldmath.h"


namespace Engine
{
	Force_GNEB::Force_GNEB(std::shared_ptr<Data::Spin_System_Chain> c) : Force(c)
	{
	}


	void Force_GNEB::Calculate(std::vector<std::vector<double>> & configurations, std::vector<std::vector<double>> & forces)
	{
		int noi = configurations.size();
		int nos = configurations[0].size()/3;
		this->isConverged = false;
		this->maxAbsComponent = 0;

		// We assume here that we receive a vector of configurations that corresponds to the vector of systems we gave the optimizer.
		//		The Optimizer shuld respect this, but there is no way to enforce it.
		// Get Energy and Effective Field of configurations
		std::vector<double> energies(configurations.size());
		for (int i = 0; i < noi; ++i)
		{
			// Calculate the Energy of the image (and store it in image)
			energies[i] = this->c->images[i]->hamiltonian->Energy(configurations[i]);
			// NEED A NICER WAY OF DOING THIS ---- THE ENERGY ETC SHOULD NOT BE UPDATED HERE, SINCE THIS MIGHT BE CALLED
			// TO CALCULATE INTERMEDIATE FORCES INSTEAD OF THE FORCES ON THE SPIN SYSTEMS
			this->c->images[i]->E = energies[i];
			// Calculate Effective Field
			//c->images[i]->Effective_Field(configurations[i], beff);
			// Calculate relevant tangent to magnetisation sphere, considering also the energies of images
			//Utility::Manifoldmath::Tangent(*c, i, t[i]);
		}

		// Calculate relevant tangent to magnetisation sphere, considering also the energies of images
		Utility::Manifoldmath::Tangents(configurations, energies, this->c->tangents);

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
			// The gradient force (unprojected) is simply the effective field
			this->c->images[img]->hamiltonian->Effective_Field(configurations[img], F_gradient[img]);
			// NEED A NICER WAY OF DOING THIS:
			this->c->images[img]->effective_field = F_gradient[img];

			// Calculate Force
			if (c->climbing_image[img])
			{
				// We reverse the component in tangent direction
				Utility::Manifoldmath::Project_Reverse(F_gradient[img], this->c->tangents[img]);
				// And Spring Force is zero
				forces[img] = F_gradient[img];
			}
			else if (c->falling_image[img])
			{
				// We project the gradient force orthogonal to the tangent
				// If anything, project orthogonal to the spins... idiot! But Heun already does that.
				//Utility::Manifoldmath::Project_Orthogonal(F_gradient[img], this->c->tangents[img]);
				// Spring Force is zero
				forces[img] = F_gradient[img];
			}
			else
			{

				// We project the gradient force orthogonal to the SPIN
				//Utility::Manifoldmath::Project_Orthogonal(F_gradient[img], this->c->tangents[img]);
				// Get the scalar product of the vectors
				double v1v2 = 0.0;
				int dim;
				// Take out component in direction of v2
				for (int i = 0; i < nos; ++i)
				{
					v1v2 = 0.0;
					for (dim = 0; dim < 3; ++dim)
					{
						v1v2 += F_gradient[img][i+dim*nos] * configurations[img][i+dim*nos];
					}
					for (dim = 0; dim < 3; ++dim)
					{
						F_gradient[img][i + dim*nos] = F_gradient[img][i + dim*nos] - v1v2 * configurations[img][i + dim*nos];
					}
				}


				// We project the gradient force orthogonal to the TANGENT
				//Utility::Manifoldmath::Project_Orthogonal(F_gradient[img], this->c->tangents[img]);
				// Get the scalar product of the vectors
				v1v2 = 0.0;
				for (int i = 0; i < 3*nos; ++i)
				{
					v1v2 += F_gradient[img][i] * this->c->tangents[img][i];
				}
				// Take out component in direction of v2
				for (int i = 0; i < 3 * nos; ++i)
				{
					F_gradient[img][i] = F_gradient[img][i] - v1v2 * this->c->tangents[img][i];
				}


				// Calculate the spring force
				//spring_forces(:, : ) = spring_constant *(dist_geodesic(NOS, IMAGES_LAST(idx_img + 1, :, : ), IMAGES(idx_img, :, : )) - dist_geodesic(NOS, IMAGES(idx_img, :, : ), IMAGES_LAST(idx_img - 1, :, : )))* tangents(:, : );
				double d1, d2, d;
				d1 = Utility::Manifoldmath::Dist_Geodesic(configurations[img + 1], configurations[img]);
				d2 = Utility::Manifoldmath::Dist_Geodesic(configurations[img], configurations[img - 1]);
				d = this->c->gneb_parameters->spring_constant * (d1 - d2);
				for (unsigned int i = 0; i < F_spring[0].size(); ++i)
				{
					F_spring[img][i] = d * this->c->tangents[img][i];
				}

				// Calculate the total force
				for (int j = 0; j < 3 * nos; ++j)
				{
					forces[img][j] = F_gradient[img][j] + F_spring[img][j];
				}
			}// end if climbing
		}// end for img=1..noi-1

		// Check for convergence
		for (int img = 1; img < noi - 1; ++img)
		{
			double fmax = this->Force_on_Image_MaxAbsComponent(configurations[img], forces[img]);
			if (fmax > this->c->gneb_parameters->force_convergence) this->isConverged = false;
			if (fmax > this->maxAbsComponent) this->maxAbsComponent = fmax;
		}
	}// end Calculate

	bool Force_GNEB::IsConverged()
	{
		return this->isConverged;
	}
}// end namespace Engine