#pragma once
#ifndef VECTORMATH_NEW_H
#define VECTORMATH_NEW_H

#include <vector>
#include <memory>

#include <Eigen/Core>

#include "Vectormath_Defines.hpp"

namespace Engine
{
	namespace Vectormath
	{

		// Returns the Bohr Magneton
		scalar MuB();
		scalar kB();

		//Prints a 1-d array of scalars to console
		void Array_to_Console(const scalar *array, const int length);
		//Prints a 1-d array of ints to console
		void Array_to_Console(const int *array, const int length);



		std::vector<scalar> scalar_product(std::vector<Vector3> vector_v1, std::vector<Vector3> vector_v2);


		void Normalize(std::vector<Vector3> & spins);

		scalar dist_greatcircle(Vector3 v1, Vector3 v2);
		scalar dist_geodesic(std::vector<Vector3> v1, std::vector<Vector3> v2);

		void Project_Reverse(std::vector<Vector3> v1, std::vector<Vector3> v2);

		void Tangents(std::vector<std::shared_ptr<std::vector<Vector3>>> configurations, std::vector<scalar> energies, std::vector<std::vector<Vector3>> & tangents);
	}
}

#endif