#pragma once
#ifndef UTILITY_MANIFOLDMATH_H
#define UTILITY_MANIFOLDMATH_H

#include "Spin_System.h"
#include "Spin_System_Chain.h"

namespace Utility
{
	namespace Manifoldmath
	{
		// Calculates the tangent in configuration space along a configuration chain at configuration idx_img
		//void Tangent(Data::Spin_System_Chain & c, int idx_img, std::vector<double> & Field);
		void Tangents(std::vector<std::shared_ptr<std::vector<double>>> configurations, std::vector<double> energies, std::vector<std::vector<double>> & tangents);

		// Normalise a 3N-dimensional vector v in 3N dimensions
		void Normalise(std::vector<double> & v);

		// Project a 3N-dimensional vector v1 orthogonal to a 3N-dim. vector v2
		void Project_Orthogonal(std::vector<double> & v1, std::vector<double> & v2);

		// Project a 3N-dimensional vector v1 into reverse directions w.r.t. a 3N-dim. vector v2
		void Project_Reverse(std::vector<double> & v1, std::vector<double> & v2);

		// Geodesic distance between two configurations
		double Dist_Geodesic(std::vector<double> s1, std::vector<double> s2);

		// Greatcircle distance between two vectors
		double Dist_Greatcircle_(std::vector<double> v1, std::vector<double> v2);

		// Greatcircle distance between two spins of two images
		double Dist_Greatcircle(std::vector<double> image1, std::vector<double> image2, int idx_spin=0);

		// Rotate a spin around an axis
		void Rotate_Spin(std::vector<double> v, std::vector<double> axis, double angle, std::vector<double> & v_out);
	}

}
#endif