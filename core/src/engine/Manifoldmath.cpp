#ifndef USE_CUDA

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
            for (unsigned int i = 0; i < vf.size(); ++i)
            {
                vf[i] *= x;
            }
        }

        void project_parallel(vectorfield & vf1, const vectorfield & vf2)
        {
            vectorfield vf3 = vf1;
            project_orthogonal(vf3, vf2);
            // TODO: replace the loop with Vectormath Kernel
            for (unsigned int i = 0; i < vf1.size(); ++i)
            {
                vf1[i] -= vf3[i];
            }
        }

        void project_orthogonal(vectorfield & vf1, const vectorfield & vf2)
        {
            scalar x = Vectormath::dot(vf1, vf2);
            // TODO: replace the loop with Vectormath Kernel
            for (unsigned int i=0; i<vf1.size(); ++i)
            {
                vf1[i] -= x*vf2[i];
            }
        }

        void invert_parallel(vectorfield & vf1, const vectorfield & vf2)
        {
            scalar x = Vectormath::dot(vf1, vf2);
            // TODO: replace the loop with Vectormath Kernel
            for (unsigned int i=0; i<vf1.size(); ++i)
            {
                vf1[i] -= 2*x*vf2[i];
            }
        }
        
        void invert_orthogonal(vectorfield & vf1, const vectorfield & vf2)
        {
            vectorfield vf3 = vf1;
            project_orthogonal(vf3, vf2);
            // TODO: replace the loop with Vectormath Kernel
            for (unsigned int i = 0; i < vf1.size(); ++i)
            {
                vf1[i] -= 2 * vf3[i];
            }
        }

        void project_tangential(vectorfield & vf1, const vectorfield & vf2)
		{
			for (unsigned int i = 0; i < vf1.size(); ++i)
			{
				vf1[i] -= vf1[i].dot(vf2[i]) * vf2[i];
			}
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
			for (unsigned int i = 0; i < v1.size(); ++i)
			{
				dist = dist + pow(dist_greatcircle(v1[i], v2[i]), 2);
			}
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

					//tangents = IMAGES_LAST(idx_img + 1, :, : ) - IMAGES_LAST(idx_img, :, : );
					for (int i = 0; i < nos; ++i)
					{
						tangents[idx_img][i] = image_plus[i] - image[i];
					}
				}
				// Last Image
				else if (idx_img == noi - 1)
				{
					auto& image_minus = *configurations[idx_img - 1];

					//tangents = IMAGES_LAST(idx_img, :, : ) - IMAGES_LAST(idx_img - 1, :, : );
					for (int i = 0; i < nos; ++i)
					{
						tangents[idx_img][i] = image[i] - image_minus[i];
					}
				}
				// Images Inbetween
				else
				{
					auto& image_plus = *configurations[idx_img + 1];
					auto& image_minus = *configurations[idx_img - 1];

					// Energies
					scalar E_mid = 0, E_plus = 0, E_minus = 0;
					E_mid = energies[idx_img];
					E_plus = energies[idx_img + 1];
					E_minus = energies[idx_img - 1];

					// Vectors to neighbouring images
					vectorfield t_plus(nos), t_minus(nos);
					for (int i = 0; i < nos; ++i)
					{
						t_plus[i] = image_plus[i] - image[i];
						t_minus[i] = image[i] - image_minus[i];
					}

					// Near maximum or minimum
					if ((E_plus < E_mid && E_mid > E_minus) || (E_plus > E_mid && E_mid < E_minus))
					{
						// Get a smooth transition between forward and backward tangent
						scalar E_max = std::fmax(std::abs(E_plus - E_mid), std::abs(E_minus - E_mid));
						scalar E_min = std::fmin(std::abs(E_plus - E_mid), std::abs(E_minus - E_mid));

						if (E_plus > E_minus)
						{
							//tangents = t_plus*E_max + t_minus*E_min;
							for (int i = 0; i < nos; ++i)
							{
								tangents[idx_img][i] = t_plus[i] * E_max + t_minus[i] * E_min;
							}
						}
						else
						{
							//tangents = t_plus*E_min + t_minus*E_max;
							for (int i = 0; i < nos; ++i)
							{
								tangents[idx_img][i] = t_plus[i] * E_min + t_minus[i] * E_max;
							}
						}
					}
					// Rising slope
					else if (E_plus > E_mid && E_mid > E_minus)
					{
						//tangents = t_plus;
						for (int i = 0; i < nos; ++i)
						{
							tangents[idx_img][i] = t_plus[i];
						}
					}
					// Falling slope
					else if (E_plus < E_mid && E_mid < E_minus)
					{
						//tangents = t_minus;
						for (int i = 0; i < nos; ++i)
						{
							tangents[idx_img][i] = t_minus[i];
						}
					}
					// No slope(constant energy)
					else
					{
						//tangents = t_plus + t_minus;
						for (int i = 0; i < nos; ++i)
						{
							tangents[idx_img][i] = t_plus[i] + t_minus[i];
						}
					}

				}

				// Project tangents into tangent planes of spin vectors to make them actual tangents
        		project_tangential(tangents[idx_img], image);

				// //Project_Orthogonal(tangents[idx_img], configurations[idx_img]);
				// scalar v1v2 = 0.0;
				// for (int i = 0; i < nos; ++i)
				// {
				// 	// Get the scalar product of the vectors
				// 	tangents[idx_img][i] -= tangents[idx_img][i].dot(image[i]) * image[i];
				// }

				// Normalise in 3N - dimensional space
				Manifoldmath::normalize(tangents[idx_img]);

			}// end for idx_img
		}// end Tangents
    }
}

#endif