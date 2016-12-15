#pragma once
#ifndef UTILITY_MANIFOLDMATH_H
#define UTILITY_MANIFOLDMATH_H

#include "Core_Defines.h"
#include "Spin_System.hpp"
#include "Spin_System_Chain.hpp"

namespace Utility
{
	namespace Manifoldmath
	{
		// Calculates the tangent in configuration space along a configuration chain at configuration idx_img
		//void Tangent(Data::Spin_System_Chain & c, int idx_img, std::vector<scalar> & Field);
		void Tangents(std::vector<std::shared_ptr<std::vector<scalar>>> configurations, std::vector<scalar> energies, std::vector<std::vector<scalar>> & tangents);

		// Normalise a 3N-dimensional vector v in 3N dimensions
		void Normalise(std::vector<scalar> & v);

		// Project a 3N-dimensional vector v1 orthogonal to a 3N-dim. vector v2
		void Project_Orthogonal(std::vector<scalar> & v1, std::vector<scalar> & v2);

		// Project a 3N-dimensional vector v1 into reverse directions w.r.t. a 3N-dim. vector v2
		void Project_Reverse(std::vector<scalar> & v1, std::vector<scalar> & v2);

		// Geodesic distance between two configurations
		scalar Dist_Geodesic(std::vector<scalar> s1, std::vector<scalar> s2);

		// Greatcircle distance between two vectors
		scalar Dist_Greatcircle_(std::vector<scalar> v1, std::vector<scalar> v2);

		// Greatcircle distance between two spins of two images
		scalar Dist_Greatcircle(std::vector<scalar> image1, std::vector<scalar> image2, int idx_spin=0);

		// Rotate a spin around an axis
		void Rotate_Spin(std::vector<scalar> v, std::vector<scalar> axis, scalar angle, std::vector<scalar> & v_out);
	}

}
#endif