#include "Configuration_Chain.h"
#include "Configurations.h"
#include "Spin_System.h"
#include "Vectormath.h"
#include "Manifoldmath.h"

#include <random>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace Utility
{
	namespace Configuration_Chain
	{
		void Add_Noise_Temperature(std::shared_ptr<Data::Spin_System_Chain> c, int idx_1, int idx_2, double temperature)
		{
			for (int img = idx_1 + 1; img <= idx_2 - 1; ++img)
			{
				Configurations::Add_Noise_Temperature(*c->images[img], temperature, img);
			}
		}

		void Homogeneous_Rotation(std::shared_ptr<Data::Spin_System_Chain> c, int idx_1, int idx_2)
		{
			int nos = c->images[0]->nos;
			int noi = idx_2 - idx_1 + 1;

			double angle, r, rot_angle;
			std::vector<double> axis(3), rot_axis(3), a(3), b(3), temp(3);

			for (int i = 0; i < nos; ++i)
			{
				for (int dim = 0; dim < 3; ++dim)
				{
					a[dim] = (*c->images[idx_1]->spins)[i + dim*nos];
					b[dim] = (*c->images[idx_2]->spins)[i + dim*nos];
				}

				r = std::fmax(-1.0, std::fmin(1.0, Vectormath::Dot_Product(a, b)));
				rot_angle = std::acos(r);
				Vectormath::Cross_Product(a, b, rot_axis);

				// If they are not strictly parallel we can rotate
				if (std::abs(Vectormath::Length(rot_axis)) > 1e-18)
				{
					Vectormath::Normalize(rot_axis);

					for (int img = idx_1; img <= idx_2; ++img)
					{
						angle = (img)*rot_angle / (noi - 1);
						Manifoldmath::Rotate_Spin(a, rot_axis, angle, temp);

						for (int dim = 0; dim < 3; ++dim)
						{
							(*c->images[img]->spins)[i + dim*nos] = temp[dim];
						}
					}
				}
				// Otherwise we simply leave the spin untouched
				else
				{
					for (int img = 1; img < noi - 1; ++img)
					{
						for (int dim = 0; dim < 3; ++dim)
						{
							(*c->images[img]->spins)[i + dim*nos] = a[dim];
						}
					}
				}
			}

		}

		void Homogeneous_Rotation(std::shared_ptr<Data::Spin_System_Chain> c, std::vector<double> A, std::vector<double> B)
		{
			(*c->images[0]->spins) = A;
			(*c->images[c->noi - 1]->spins) = B;

			int nos = c->images[0]->nos;

			double angle, r, rot_angle;
			std::vector<double> axis(3), rot_axis(3), a(3), b(3), temp(3);

			for (int i = 0; i < c->images[0]->nos; ++i)
			{
				for (int dim = 0; dim < 3; ++dim)
				{
					a[dim] = A[i + dim*nos];
					b[dim] = B[i + dim*nos];
				}
				
				r = std::fmax(-1.0, std::fmin(1.0, Vectormath::Dot_Product(a, b)));
				rot_angle = std::acos(r);
				Vectormath::Cross_Product(a, b, rot_axis);

				// If they are not strictly parallel we can rotate
				if (std::abs(Vectormath::Length(rot_axis)) > 1e-18)
				{
					Vectormath::Normalize(rot_axis);

					for (int img = 1; img < c->noi - 1; ++img)
					{
						angle = (img)*rot_angle / (c->noi - 1);
						Manifoldmath::Rotate_Spin(a, rot_axis, angle, temp);

						for (int dim = 0; dim < 3; ++dim)
						{
							(*c->images[img]->spins)[i + dim*nos] = temp[dim];
						}
					}
				}
				// Otherwise we simply leave the spin untouched
				else
				{
					for (int img = 1; img < c->noi - 1; ++img)
					{
						for (int dim = 0; dim < 3; ++dim)
						{
							(*c->images[img]->spins)[i + dim*nos] = a[dim];
						}
					}
				}
			}

			/*
				do i=1,NOS
				  Start(i,:)  = Start(i,:)/length(Start(i,:))
				  Finish(i,:) = Finish(i,:)/length(Finish(i,:))
      
				  IMAGES(idx_start,i,:)  = Start(i,:)
				  IMAGES(idx_finish,i,:) = Finish(i,:)
      
				  r = max(-1.0,min(1.0, dot_product(Start(i,:), Finish(i,:)) )) !! this prevents NaNs from ocurring
				  rot_angle(i)  = acos(r)
				  rot_axis(i,:) = cross_product(Start(i,:), Finish(i,:))
				  if (abs(length(rot_axis(i,:))) > 1e-18) rot_axis(i,:) = rot_axis(i,:)/length(rot_axis(i,:))
				enddo
			*/

			/*
				do i=idx_start+1,idx_finish-1
				  !! loop over spins in image
				  do j=1,NOS
					 angle = (i-1)*rot_angle(j)/(idx_finish-idx_start)
					 IMAGES(i,j,:) = rotate_spin(Start(j,:),rot_axis(j,:),angle)
         
				  enddo
				enddo
			*/
		}
	}//end namespace Configuration_Chain
}//end namespace Utility