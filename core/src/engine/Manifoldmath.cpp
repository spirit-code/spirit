#ifndef SPIRIT_USE_CUDA

#include <engine/Vectormath.hpp>
#include <engine/Manifoldmath.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>

#include <Eigen/Dense>

#include <array>

namespace Engine
{
	namespace Manifoldmath
	{
        scalar norm(const vectorfield & vf)
        {
            scalar x = Vectormath::dot(vf, vf);
            return std::sqrt(x);
        }

        void normalize(vectorfield & vf)
        {
            scalar x = 1.0/norm(vf);
            #pragma omp parallel for
            for (unsigned int i = 0; i < vf.size(); ++i)
                vf[i] *= x;
        }

        void project_parallel(vectorfield & vf1, const vectorfield & vf2)
        {
            vectorfield vf3 = vf1;
            project_orthogonal(vf3, vf2);
            // TODO: replace the loop with Vectormath Kernel
            #pragma omp parallel for
            for (unsigned int i = 0; i < vf1.size(); ++i)
                vf1[i] -= vf3[i];
        }

        void project_orthogonal(vectorfield & vf1, const vectorfield & vf2)
        {
            scalar x = Vectormath::dot(vf1, vf2);
            // TODO: replace the loop with Vectormath Kernel
            #pragma omp parallel for
            for (unsigned int i=0; i<vf1.size(); ++i)
                vf1[i] -= x*vf2[i];
        }

        void invert_parallel(vectorfield & vf1, const vectorfield & vf2)
        {
            scalar x = Vectormath::dot(vf1, vf2);
            // TODO: replace the loop with Vectormath Kernel
            #pragma omp parallel for
            for (unsigned int i=0; i<vf1.size(); ++i)
                vf1[i] -= 2*x*vf2[i];
        }
        
        void invert_orthogonal(vectorfield & vf1, const vectorfield & vf2)
        {
            vectorfield vf3 = vf1;
            project_orthogonal(vf3, vf2);
            // TODO: replace the loop with Vectormath Kernel
            #pragma omp parallel for
            for (unsigned int i = 0; i < vf1.size(); ++i)
                vf1[i] -= 2 * vf3[i];
        }

        void project_tangential(vectorfield & vf1, const vectorfield & vf2)
		{
            #pragma omp parallel for
			for (unsigned int i = 0; i < vf1.size(); ++i)
				vf1[i] -= vf1[i].dot(vf2[i]) * vf2[i];
		}


		scalar dist_greatcircle(const Vector3 & v1, const Vector3 & v2)
		{
			scalar r = v1.dot(v2);

			// Prevent NaNs from occurring
			r = std::fmax(-1.0, std::fmin(1.0, r));

			// Greatcircle distance
			return std::acos(r);
		}


		scalar dist_geodesic(const vectorfield & v1, const vectorfield & v2)
		{
			scalar dist = 0;
            #pragma omp parallel for reduction(+:dist)
			for (unsigned int i = 0; i < v1.size(); ++i)
				dist += pow(dist_greatcircle(v1[i], v2[i]), 2);
			return sqrt(dist);
		}

		/*
		Calculates the 'tangent' vectors, i.e.in crudest approximation the difference between an image and the neighbouring
		*/
		void Tangents(std::vector<std::shared_ptr<vectorfield>> configurations, const std::vector<scalar> & energies, std::vector<vectorfield> & tangents)
		{
			int noi = configurations.size();
			int nos = (*configurations[0]).size();

			for (int idx_img = 0; idx_img < noi; ++idx_img)
			{
				auto& image = *configurations[idx_img];

				// First Image
				if (idx_img == 0)
				{
					auto& image_plus = *configurations[idx_img + 1];
					Vectormath::set_c_a( 1, image_plus, tangents[idx_img]);
					Vectormath::add_c_a(-1, image,      tangents[idx_img]);
				}
				// Last Image
				else if (idx_img == noi - 1)
				{
					auto& image_minus = *configurations[idx_img - 1];
					Vectormath::set_c_a( 1, image,       tangents[idx_img]);
					Vectormath::add_c_a(-1, image_minus, tangents[idx_img]);
				}
				// Images Inbetween
				else
				{
					auto& image_plus  = *configurations[idx_img + 1];
					auto& image_minus = *configurations[idx_img - 1];

					// Energies
					scalar E_mid = 0, E_plus = 0, E_minus = 0;
					E_mid   = energies[idx_img];
					E_plus  = energies[idx_img + 1];
					E_minus = energies[idx_img - 1];

					// Vectors to neighbouring images
					vectorfield t_plus(nos), t_minus(nos);

					Vectormath::set_c_a( 1, image_plus, t_plus);
					Vectormath::add_c_a(-1, image,      t_plus);

					Vectormath::set_c_a( 1, image,       t_minus);
					Vectormath::add_c_a(-1, image_minus, t_minus);

					// Near maximum or minimum
					if ((E_plus < E_mid && E_mid > E_minus) || (E_plus > E_mid && E_mid < E_minus))
					{
						// Get a smooth transition between forward and backward tangent
						scalar E_max = std::max(std::abs(E_plus - E_mid), std::abs(E_minus - E_mid));
						scalar E_min = std::min(std::abs(E_plus - E_mid), std::abs(E_minus - E_mid));

						if (E_plus > E_minus)
						{
							Vectormath::set_c_a(E_max, t_plus,  tangents[idx_img]);
							Vectormath::add_c_a(E_min, t_minus, tangents[idx_img]);
						}
						else
						{
							Vectormath::set_c_a(E_min, t_plus,  tangents[idx_img]);
							Vectormath::add_c_a(E_max, t_minus, tangents[idx_img]);
						}
					}
					// Rising slope
					else if (E_plus > E_mid && E_mid > E_minus)
					{
						Vectormath::set_c_a(1, t_plus,  tangents[idx_img]);
					}
					// Falling slope
					else if (E_plus < E_mid && E_mid < E_minus)
					{
						Vectormath::set_c_a(1, t_minus,  tangents[idx_img]);
						//tangents = t_minus;
						for (int i = 0; i < nos; ++i)
						{
							tangents[idx_img][i] = t_minus[i];
						}
					}
					// No slope(constant energy)
					else
					{
						Vectormath::set_c_a(1, t_plus,  tangents[idx_img]);
						Vectormath::add_c_a(1, t_minus, tangents[idx_img]);
					}
				}

				// Project tangents into tangent planes of spin vectors to make them actual tangents
        		project_tangential(tangents[idx_img], image);

				// Normalise in 3N - dimensional space
				Manifoldmath::normalize(tangents[idx_img]);

			}// end for idx_img
		}// end Tangents
    }
}

#endif