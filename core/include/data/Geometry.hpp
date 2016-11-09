#pragma once
#ifndef DATA_GEOMETRY_H
#define DATA_GEOMETRY_H

#include "Core_Defines.h"

#include <vector>
#include <array>

namespace Data
{
    // TODO: replace that type with Eigen!
    typedef struct {
        scalar x, y, z;
    } vector_t;

    typedef std::array<int, 4> tetrahedron_t;

	// Geometry contains all geometric information about the spin_system
	// The members are const, as a spin system has to be created new whenever one of these members is changed.
	class Geometry
	{

	public:
		// Constructor
		Geometry(const std::vector<std::vector<scalar>> basis, const std::vector<std::vector<scalar>> translation_vectors,
			const std::vector<int> n_cells, const int n_spins_basic_domain, const std::vector<std::vector<scalar>> basis_atoms,
			const std::vector<scalar> spin_pos);
		// Destructor
		//~Geometry();


		// -------------------- Input constants ------------------
		// Basis [dim][basis_vec]
		const std::vector<std::vector<scalar>> basis;
		// Translation Vectors [dim][transl_vec]
		const std::vector<std::vector<scalar>> translation_vectors;
		// Number of Translations {nta, ntb, ntc}
		const std::vector<int> n_cells;
		// Number of spins per basic domain
		const int n_spins_basic_domain;
		// Array of basis atom positions [3][n_basis_atoms]
		std::vector<std::vector<scalar>> basis_atoms;
		// Number of Spins total
		const int nos;
		// number of shells -> moved to Hamiltonian
		// const int n_shells;

		// Center and Bounds
		std::vector<scalar> center;
		std::vector<scalar> bounds_min;
		std::vector<scalar> bounds_max;

		// -------------------- Calculated Geometry ------------------
		// number of boundary vectors
		//const int number_boundary_vectors;
		// Boundary vectors: boundary_vectors[dim][number_b_vectors]
		//const std::vector<std::vector<scalar>> boundary_vectors;

		// Positions of the Spins: spin_pos[dim][nos]
		const std::vector<scalar> spin_pos;

		// //////  What are the segments used for??
		// segments[nos][4]
		//const std::vector<std::vector<int>> segments;
		// Position of the Segments: segments_pos[dim][nos][4]
    //const std::vector<std::vector<std::vector<scalar>>> segments_pos;

    const std::vector<tetrahedron_t>& triangulation();

    bool is2D() const;
    
  private:
    std::vector<tetrahedron_t> _triangulation;
	};
}
#endif
