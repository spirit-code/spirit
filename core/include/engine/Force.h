#pragma once
#ifndef FORCE_H
#define FORCE_H

#include <vector>
#include <algorithm>
#include <iterator>

#include "Spin_System.h"
#include "Spin_System_Chain.h"

//namespace Data
//{
//	class Spin_System_Chain;
//}

namespace Engine
{
	class Force
	{
	public:
		Force(std::shared_ptr<Data::Spin_System_Chain> c)
		{
			this->c = c;
		}

		virtual void Calculate(std::vector<std::vector<double>> & configurations, std::vector<std::vector<double>> & forces) {}; // LLG force should check for convergence of single images

		virtual bool IsConverged() { return false; }

		double maxAbsComponent;

	protected:
		std::shared_ptr<Data::Spin_System_Chain> c;


		// Return the maximum of absolute values of force components for an image
		double Force_on_Image_MaxAbsComponent(std::vector<double> & image, std::vector<double> force)
		{
			int nos = image.size()/3;
			// We project the force orthogonal to the SPIN
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
					v1v2 += force[i + dim*nos] * image[i + dim*nos];
				}
				for (dim = 0; dim < 3; ++dim)
				{
					force[i + dim*nos] = force[i + dim*nos] - v1v2 * image[i + dim*nos];
				}
			}

			// We want the Maximum of Absolute Values of all force components on all images
			double absmax = 0;
			// Find minimum and maximum values
			auto minmax = std::minmax_element(force.begin(), force.end());
			// Mamimum of absolute values
			absmax = std::max(absmax, std::abs(*minmax.first));
			absmax = std::max(absmax, std::abs(*minmax.second));
			// Return
			return absmax;
		}
	};
}
#endif