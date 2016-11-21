#pragma once
#ifndef VECTORMATH_NEW_H
#define VECTORMATH_NEW_H

#include <vector>

#include <Eigen/Core>

#include "Vectormath_Defines.hpp"

namespace Engine
{
	namespace Vectormath
	{
		std::vector<scalar> scalar_product(std::vector<Vector3> vector_v1, std::vector<Vector3> vector_v2);


		void Normalize_3Nos(std::vector<Vector3> & spins);

		scalar Engine::Vectormath::dist_greatcircle(Vector3 v1, Vector3 v2);
	}
}

#endif