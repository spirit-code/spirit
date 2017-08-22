#pragma once
#ifndef DATA_GEOMETRY_H
#define DATA_GEOMETRY_H

#include "Spirit_Defines.h"
#include <engine/Vectormath_Defines.hpp>

#include <vector>

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
        Geometry(std::vector<Vector3> basis, std::vector<Vector3> translation_vectors,
            intfield n_cells, std::vector<Vector3> basis_atoms, scalar lattice_constant,
            vectorfield spin_pos, intfield atom_types);
        // Destructor
        //~Geometry();

        const std::vector<tetrahedron_t>& triangulation(int n_cell_step=1);
        int calculateDimensionality() const;

        // -------------------- Input constants ------------------
        // Basis [dim][basis_vec]
        std::vector<Vector3> basis;
        // Lattice Constant [Angstrom]
        scalar lattice_constant;
        // Translation Vectors [dim][transl_vec]
        std::vector<Vector3> translation_vectors;
        // Number of Translations {nta, ntb, ntc}
        intfield n_cells;
        // Number of spins per basic domain
        int n_spins_basic_domain;
        // Array of basis atom positions [3][n_basis_atoms]
        std::vector<Vector3> basis_atoms;
        // Number of Spins total
        int nos;

        // Center and Bounds
        Vector3 center, bounds_min, bounds_max;
        // Unit Cell Bounds
        Vector3 cell_bounds_min, cell_bounds_max;

        // -------------------- Calculated Geometry ------------------
        // number of boundary vectors
        //const int number_boundary_vectors;
        // Boundary vectors: boundary_vectors[dim][number_b_vectors]
        //const std::vector<std::vector<scalar>> boundary_vectors;

        // Positions of the Spins: spin_pos[dim][nos]
        vectorfield spin_pos;

        // Atom types: type index 0..n or or vacancy (type < 0)
        intfield atom_types;

        // //////  What are the segments used for??
        // segments[nos][4]
        //const std::vector<std::vector<int>> segments;
        // Position of the Segments: segments_pos[dim][nos][4]
        //const std::vector<std::vector<std::vector<scalar>>> segments_pos;

        int dimensionality;
        
    private:
        std::vector<tetrahedron_t> _triangulation;
    };
}
#endif
