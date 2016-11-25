#pragma once
#ifndef VECTORMATH_NEW_H
#define VECTORMATH_NEW_H

#include <vector>
#include <memory>

#include <Eigen/Core>

#include <engine/Vectormath_Defines.hpp>

namespace Engine
{
	namespace Vectormath
	{

		// Returns the Bohr Magneton
		scalar MuB();
		scalar kB();

		void Build_Spins(std::vector<Vector3> & spin_pos, std::vector<Vector3> & basis_atoms, std::vector<Vector3> & translation_vectors, std::vector<int> & n_cells, const int nos_basic);


		std::vector<scalar> scalar_product(const std::vector<Vector3> & vector_v1, const std::vector<Vector3> & vector_v2);


		void Normalize(std::vector<Vector3> & spins);

		scalar dist_greatcircle(const Vector3 & v1, const Vector3 & v2);
		scalar dist_geodesic(const std::vector<Vector3> & v1, const std::vector<Vector3> & v2);

		void Project_Reverse(std::vector<Vector3> & v1, const std::vector<Vector3> & v2);

		void Rotate_Spin(const Vector3 & v, const Vector3 & axis, const scalar & angle, Vector3 & v_out);

		void Tangents(std::vector<std::shared_ptr<std::vector<Vector3>>> configurations, const std::vector<scalar> & energies, std::vector<std::vector<Vector3>> & tangents);
	}
}

#endif