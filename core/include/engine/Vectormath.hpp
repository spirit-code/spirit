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
		/////////////////////////////////////////////////////////////////
		//////// Regular Math
		
		// Returns the Bohr Magneton [meV / T]
		scalar MuB();
		// Returns the Boltzmann constant [meV / K]
		scalar kB();



		/////////////////////////////////////////////////////////////////
		//////// Single Vector Math

		// Rotate a vector around an axis by a certain degree
		void rotate(const Vector3 & v, const Vector3 & axis, const scalar & angle, Vector3 & v_out);


		/////////////////////////////////////////////////////////////////
		//////// Vectorfield Math - special stuff

		// Build an array of spin positions
		void Build_Spins(vectorfield & spin_pos, std::vector<Vector3> & basis_atoms, std::vector<Vector3> & translation_vectors, std::vector<int> & n_cells, const int nos_basic);
		// Calculate the mean of a vectorfield
		std::array<scalar, 3> Magnetization(const vectorfield & vf);


		/////////////////////////////////////////////////////////////////
		//////// Vectormath-like operations

		// sets sf := s
		// sf is a scalarfield
		// s is a scalar
		void fill(scalarfield & sf, scalar s);

		// Scale a scalarfield by a given value
		void scale(scalarfield & sf, scalar s);

		// Sum over a scalarfield
		scalar sum(const scalarfield & sf);

		// Calculate the mean of a scalarfield
		scalar mean(const scalarfield & sf);

		// sets vf := v
		// vf is a vectorfield
		// v is a vector
		void fill(vectorfield & vf, const Vector3 & v);

		// Scale a vectorfield by a given value
		void scale(vectorfield & vf, const scalar & sc);

		// Sum over a vectorfield
		Vector3 sum(const vectorfield & vf);

		// Calculate the mean of a vectorfield
		Vector3 mean(const vectorfield & vf);


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
	}
}

#endif