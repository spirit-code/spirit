#include "Configuration_Chain.hpp"
#include "Configurations.hpp"
#include "Spin_System.hpp"
#include "engine/Vectormath.hpp"
#include "Manifoldmath.hpp"

#include <Eigen/Dense>

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
		void Add_Noise_Temperature(std::shared_ptr<Data::Spin_System_Chain> c, int idx_1, int idx_2, scalar temperature)
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

			scalar angle, rot_angle;
			Vector3 axis, rot_axis, a, b, temp;

			for (int i = 0; i < nos; ++i)
			{
				a = (*c->images[idx_1]->spins)[i];
				b = (*c->images[idx_2]->spins)[i];

				rot_angle = Engine::Vectormath::dist_greatcircle(a, b);
				rot_axis = a.cross(b);

				// If they are not strictly parallel we can rotate
				if (rot_axis.norm() > 1e-8)
				{
					rot_axis.normalize();

					for (int img = idx_1; img <= idx_2; ++img)
					{
						angle = (img)*rot_angle / (noi - 1);
						Engine::Vectormath::Rotate_Spin(a, rot_axis, angle, temp);

						(*c->images[img]->spins)[i] = temp;
					}
				}
				// Otherwise we simply leave the spin untouched
				else
				{
					for (int img = 1; img < noi - 1; ++img)
					{
						(*c->images[img]->spins)[i] = a;
					}
				}
			}

		}

		void Homogeneous_Rotation(std::shared_ptr<Data::Spin_System_Chain> c, std::vector<Vector3> A, std::vector<Vector3> B)
		{
			(*c->images[0]->spins) = A;
			(*c->images[c->noi - 1]->spins) = B;

			int nos = c->images[0]->nos;

			scalar angle, rot_angle;
			Vector3 axis, rot_axis, a, b, temp;

			for (int i = 0; i < c->images[0]->nos; ++i)
			{
				a = A[i];
				b = B[i];
				
				rot_angle = Engine::Vectormath::dist_greatcircle(a, b);
				rot_axis = a.cross(b);

				// If they are not strictly parallel we can rotate
				if (rot_axis.norm() > 1e-8)
				{
					rot_axis.normalize();

					for (int img = 1; img < c->noi - 1; ++img)
					{
						angle = (img)*rot_angle / (c->noi - 1);
						Engine::Vectormath::Rotate_Spin(a, rot_axis, angle, temp);

						(*c->images[img]->spins)[i] = temp;
					}
				}
				// Otherwise we simply leave the spin untouched
				else
				{
					for (int img = 1; img < c->noi - 1; ++img)
					{
						(*c->images[img]->spins)[i] = a;
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