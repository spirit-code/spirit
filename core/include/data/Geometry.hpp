#pragma once
#ifndef DATA_GEOMETRY_H
#define DATA_GEOMETRY_H

#include "Spirit_Defines.h"
#include <engine/Vectormath_Defines.hpp>
#include <Spirit/Geometry.h>

#include <vector>

namespace Data
{
    // TODO: replace that type with Eigen!
    typedef struct {
        scalar x, y, z;
    } vector3_t;
    typedef struct {
        scalar x, y;
    } vector2_t;

    typedef std::array<int, 4> tetrahedron_t;
    typedef std::array<int, 3> triangle_t;


    enum class BravaisLatticeType
    {
        Irregular   = Bravais_Lattice_Irregular,   // Arbitrary bravais vectors
        Rectilinear = Bravais_Lattice_Rectilinear, // Rectilinear (orthorombic) lattice
        SC          = Bravais_Lattice_SC,          // Simple cubic lattice
        Hex2D       = Bravais_Lattice_Hex2D,       // 2D Hexagonal lattice
        HCP         = Bravais_Lattice_HCP,         // Hexagonal closely packed
        BCC         = Bravais_Lattice_BCC,         // Body-centered cubic
        FCC         = Bravais_Lattice_FCC          // Face-centered cubic
    };

    // Geometry contains all geometric information of a system
    class Geometry
    {
    public:
        // ---------- Constructor
        //  Build a regular lattice from a defined basis cell and translations
        Geometry(std::vector<Vector3> bravais_vectors, intfield n_cells, std::vector<Vector3> cell_atoms, scalarfield mu_s,
            intfield cell_atom_types, scalar lattice_constant);


        // ---------- Convenience functions
        // Retrieve triangulation, if 2D
        const std::vector<triangle_t>&    triangulation(int n_cell_step=1);
        // Retrieve tetrahedra, if 3D
        const std::vector<tetrahedron_t>& tetrahedra(int n_cell_step=1);
        // Introduce disorder into the atom types
        // void disorder(scalar mixing);
        static std::vector<Vector3> BravaisVectorsSC();
        static std::vector<Vector3> BravaisVectorsFCC();
        static std::vector<Vector3> BravaisVectorsBCC();
        static std::vector<Vector3> BravaisVectorsHex2D60();
        static std::vector<Vector3> BravaisVectorsHex2D120();


        // ---------- Basic information set, which (in theory) defines everything

        // Basis vectors {a, b, c} of the unit cell
        std::vector<Vector3> bravais_vectors;
        // Lattice Constant [Angstrom] (scales the translations)
        scalar lattice_constant;
        // Number of cells {na, nb, nc}
        intfield n_cells;
        // Number of spins per basic domain
        int n_cell_atoms;
        // Array of basis atom positions
        std::vector<Vector3> cell_atoms;
        // Spin moments of basis cell atoms
        scalarfield mu_s;
        // Atom types of the atoms in a unit cell:
        // type index 0..n or or vacancy (type < 0)
        intfield cell_atom_types;


        // ---------- Inferrable information

        // The kind of geometry
        BravaisLatticeType classifier;

        // Number of Spins total
        int nos;
        // Number of basis cells total
        int n_cells_total;
        // Positions of all the atoms
        vectorfield positions;
        // Atom types of all the atoms: type index 0..n or or vacancy (type < 0)
        intfield atom_types;

        // Dimensionality of the points
        int dimensionality;
        // Center and Bounds
        Vector3 center, bounds_min, bounds_max;
        // Unit Cell Bounds
        Vector3 cell_bounds_min, cell_bounds_max;

        
    private:
        // Calculate and update the dimensionality of the points in this geometry
        void calculateDimensionality();
        // Calculate and update bounds of the System
        void calculateBounds();
		// Calculate and update unit cell bounds
		void calculateUnitCellBounds();
		// Calculate and update the type lattice
		void calculateGeometryType();

        // 
        std::vector<triangle_t>    _triangulation;
        std::vector<tetrahedron_t> _tetrahedra;
        
        // Temporaries to tell wether the triangulation or tetrahedra
        // need to be updated when the corresponding function is called
        int last_update_n_cell_step;
        intfield last_update_n_cells;
    };
}
#endif
