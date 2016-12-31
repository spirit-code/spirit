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

		void Build_Spins(vectorfield & spin_pos, std::vector<Vector3> & basis_atoms, std::vector<Vector3> & translation_vectors, std::vector<int> & n_cells, const int nos_basic);

		std::array<scalar,3> Magnetization(const vectorfield & vf);


		/////////////////////////////////////////////////////////////////

		// sets vf := v
		// vf is a vectorfield
		// v is a vector
		void fill(vectorfield & vf, const Vector3 & v);

		// Scale a vectorfield by a given value
		void scale(vectorfield & vf, const scalar & sc);

		// TODO: move this function to manifold??
		// computes the inner product of two vectorfields v1 and v2
		scalar dot(const vectorfield & vf1, const vectorfield & vf2);

		// computes the inner products of vectors in v1 and v2
		// v1 and v2 are vectorfields
		void dot(const vectorfield & vf1, const vectorfield & vf2, scalarfield & out);
		
		// computes the vector (cross) products of vectors in v1 and v2
		// v1 and v2 are vector fields
		void cross(const vectorfield & vf1, const vectorfield & vf2, vectorfield & out);
		
		// out[i] += c*a
		void add_c_a(const scalar & c, const Vector3 & a, vectorfield & out);

		// out[i] += c*a[i]
		void add_c_a(const scalar & c, const vectorfield & a, vectorfield & out);

		// out[i] += c * a*b[i]
		void add_c_dot(const scalar & c, const Vector3 & a, const vectorfield & b, scalarfield & out);

		// out[i] += c * a[i]*b[i]
		void add_c_dot(const scalar & c, const vectorfield & a, const vectorfield & b, scalarfield & out);

		// out[i] += c * a x b[i]
		void add_c_cross(const scalar & c, const Vector3 & a, const vectorfield & b, vectorfield & out);

		// out[i] += c * a[i] x b[i]
		void add_c_cross(const scalar & c, const vectorfield & a, const vectorfield & b, vectorfield & out);
		
		/////////////////////////////////////////////////////////////////


		std::vector<scalar> scalar_product(const vectorfield & vector_v1, const vectorfield & vector_v2);


		void Normalize(vectorfield & spins);

		scalar dist_greatcircle(const Vector3 & v1, const Vector3 & v2);
		scalar dist_geodesic(const vectorfield & v1, const vectorfield & v2);

		void Project_Reverse(vectorfield & v1, const vectorfield & v2);

		void Rotate_Spin(const Vector3 & v, const Vector3 & axis, const scalar & angle, Vector3 & v_out);

		void Tangents(std::vector<std::shared_ptr<vectorfield>> configurations, const std::vector<scalar> & energies, std::vector<vectorfield> & tangents);
	}
}

#endif