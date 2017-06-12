#pragma once
#ifndef VECTORMATH_NEW_H
#define VECTORMATH_NEW_H

#include <vector>
#include <memory>

#include <Eigen/Core>

#include <data/Spin_System.hpp>
#include <engine/Vectormath_Defines.hpp>

namespace Engine
{
	namespace Vectormath
	{
		/////////////////////////////////////////////////////////////////
		//////// Single Vector Math

		// Rotate a vector around an axis by a certain degree
		void rotate(const Vector3 & v, const Vector3 & axis, const scalar & angle, Vector3 & v_out);

		// Decompose a vector into numbers of translations in a basis
		Vector3 decompose(const Vector3 & v, const std::vector<Vector3> & basis);

		/////////////////////////////////////////////////////////////////
		//////// Translating across the lattice
		#ifndef USE_CUDA
		inline bool boundary_conditions_fulfilled(const intfield & n_cells, const intfield & boundary_conditions, const std::array<int, 3> & translations_i, const std::array<int, 3> & translations_j)
		{
			int da = translations_i[0] + translations_j[0];
			int db = translations_i[1] + translations_j[1];
			int dc = translations_i[2] + translations_j[2];
			return ((boundary_conditions[0] || (0 <= da && da < n_cells[0])) &&
				(boundary_conditions[1] || (0 <= db && db < n_cells[1])) &&
				(boundary_conditions[2] || (0 <= dc && dc < n_cells[2])));
		}

		inline int idx_from_translations(const intfield & n_cells, const int n_spins_basic_domain, const std::array<int, 3> & translations)
		{
			int Na = n_cells[0];
			int Nb = n_cells[1];
			int Nc = n_cells[2];
			int N = n_spins_basic_domain;

			int da = translations[0];
			int db = translations[1];
			int dc = translations[2];

			return da*N + db*N*Na + dc*N*Na*Nb;
		}

		inline int idx_from_translations(const intfield & n_cells, const int n_spins_basic_domain, const std::array<int, 3> & translations_i, const std::array<int, 3> translations)
		{
			int Na = n_cells[0];
			int Nb = n_cells[1];
			int Nc = n_cells[2];
			int N = n_spins_basic_domain;

			int da = translations_i[0] + translations[0];
			int db = translations_i[1] + translations[1];
			int dc = translations_i[2] + translations[2];

			if (translations[0] < 0)
				da += N*Na;
			if (translations[1] < 0)
				db += N*Na*Nb;
			if (translations[2] < 0)
				dc += N*Na*Nb*Nc;

			int idx = (da%Na)*N + (db%Nb)*N*Na + (dc%Nc)*N*Na*Nb;

			return idx;
		}

		inline std::array<int, 3> translations_from_idx(const intfield & n_cells, const int n_spins_basic_domain, int idx)
		{
			std::array<int, 3> ret;
			int Na = n_cells[0];
			int Nb = n_cells[1];
			int Nc = n_cells[2];
			int N = n_spins_basic_domain;
			ret[2] = idx / (N*Na*Nb);
			ret[1] = (idx - ret[2] * N*Na*Nb) / (N*Na);
			ret[0] = idx - ret[2] * N*Na*Nb - ret[1] * N*Na;
			return ret;
		}
		#endif

		/////////////////////////////////////////////////////////////////
		//////// Vectorfield Math - special stuff

		// Build an array of spin positions
		void Build_Spins(vectorfield & spin_pos, const std::vector<Vector3> & basis_atoms, const std::vector<Vector3> & translation_vectors, const intfield & n_cells);
		// Calculate the mean of a vectorfield
		std::array<scalar, 3> Magnetization(const vectorfield & vf);
		// Calculate the topological charge inside a vectorfield
		scalar TopologicalCharge(const vectorfield & vf);

		// Utility function for the SIB Optimizer - maybe create a MathUtil namespace?
		void transform(const vectorfield & spins, const vectorfield & force, vectorfield & out);
		void get_random_vectorfield(const Data::Spin_System & sys, scalar epsilon, vectorfield & xi);


		/////////////////////////////////////////////////////////////////
		//////// Vectormath-like operations

		// sets sf := s
		// sf is a scalarfield
		// s is a scalar
		void fill(scalarfield & sf, scalar s);
		void fill(scalarfield & sf, scalar s, const intfield & mask);

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
		void fill(vectorfield & vf, const Vector3 & v, const intfield & mask);

		// Normalize the vectors of a vectorfield
		void normalize_vectors(vectorfield & vf);

    // Pair of Minimum and Maximum of any component of any vector of a vectorfield
    std::pair<scalar, scalar> minmax_component(const vectorfield & v1);

		// Maximum absolute component of a vectorfield
    	scalar max_abs_component(const vectorfield & vf);

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

		// out[i] = c*a
		void set_c_a(const scalar & c, const Vector3 & a, vectorfield & out);
		void set_c_a(const scalar & c, const Vector3 & a, vectorfield & out, const intfield & mask);
		// out[i] = c*a[i]
		void set_c_a(const scalar & c, const vectorfield & a, vectorfield & out);
		void set_c_a(const scalar & c, const vectorfield & a, vectorfield & out, const intfield & mask);

		// out[i] += c * a*b[i]
		void add_c_dot(const scalar & c, const Vector3 & a, const vectorfield & b, scalarfield & out);
		// out[i] += c * a[i]*b[i]
		void add_c_dot(const scalar & c, const vectorfield & a, const vectorfield & b, scalarfield & out);
		
		// out[i] = c * a*b[i]
		void set_c_dot(const scalar & c, const Vector3 & a, const vectorfield & b, scalarfield & out);
		// out[i] = c * a[i]*b[i]
		void set_c_dot(const scalar & c, const vectorfield & a, const vectorfield & b, scalarfield & out);

		// out[i] += c * a x b[i]
		void add_c_cross(const scalar & c, const Vector3 & a, const vectorfield & b, vectorfield & out);
		// out[i] += c * a[i] x b[i]
		void add_c_cross(const scalar & c, const vectorfield & a, const vectorfield & b, vectorfield & out);
		
		// out[i] = c * a x b[i]
		void set_c_cross(const scalar & c, const Vector3 & a, const vectorfield & b, vectorfield & out);
		// out[i] = c * a[i] x b[i]
		void set_c_cross(const scalar & c, const vectorfield & a, const vectorfield & b, vectorfield & out);
	}
}

#endif