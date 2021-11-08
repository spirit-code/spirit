#pragma once
#ifndef SPIRIT_CORE_DATA_GEOMETRY_HPP
#define SPIRIT_CORE_DATA_GEOMETRY_HPP

#include "Spirit_Defines.h"
#include <Spirit/Geometry.h>
#include <engine/Vectormath_Defines.hpp>

#include <vector>

namespace Data
{

// TODO: replace that type with Eigen!
struct vector3_t
{
    double x, y, z;
};
struct vector2_t
{
    double x, y;
};

using tetrahedron_t = std::array<int, 4>;
using triangle_t    = std::array<int, 3>;

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

struct Pinning
{
    // Boundary pinning
    int na_left, na_right;
    int nb_left, nb_right;
    int nc_left, nc_right;
    vectorfield pinned_cell;

    // Individual pinned atoms
    field<Site> sites;
    // Their fixed orientation
    vectorfield spins;
};

struct Defects
{
    field<Site> sites;
    intfield types;
};

// Vector sizes: N_basis * N_atom_tyes
struct Basis_Cell_Composition
{
    bool disordered;

    // Indexing for the following information
    std::vector<int> iatom;

    // Atom types of the atoms in a unit cell:
    // type index 0..n or or vacancy (type < 0)
    std::vector<int> atom_type;

    // Magnetic moment of an atom in mu_B
    std::vector<scalar> mu_s;

    // Chemical concentration of an atom on a specific lattice site (if disorder is activated)
    std::vector<scalar> concentration;
};

// Geometry contains all geometric information of a system
class Geometry
{
public:
    // ---------- Constructor
    //  Build a regular lattice from a defined basis cell and translations
    Geometry(
        std::vector<Vector3> bravais_vectors, intfield n_cells, std::vector<Vector3> cell_atoms,
        Basis_Cell_Composition cell_composition, scalar lattice_constant, Pinning pinning, Defects defects );

    // ---------- Convenience functions
    // Retrieve triangulation, if 2D
    const std::vector<triangle_t> &
    triangulation( int n_cell_step = 1, std::array<int, 6> ranges = { 0, -1, 0, -1, 0, -1 } );
    // Retrieve tetrahedra, if 3D
    const std::vector<tetrahedron_t> &
    tetrahedra( int n_cell_step = 1, std::array<int, 6> ranges = { 0, -1, 0, -1, 0, -1 } );
    // Introduce disorder into the atom types
    // void disorder(scalar mixing);
    static std::vector<Vector3> BravaisVectorsSC();
    static std::vector<Vector3> BravaisVectorsFCC();
    static std::vector<Vector3> BravaisVectorsBCC();
    static std::vector<Vector3> BravaisVectorsHex2D60();
    static std::vector<Vector3> BravaisVectorsHex2D120();
    // Pinning
    void Apply_Pinning( vectorfield & vf );

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
    // Composition of the basis cell (atom types, mu_s, disorder)
    Basis_Cell_Composition cell_composition;
    // Info on pinned spins
    Pinning pinning;
    // Info on defects
    Defects defects;

    // ---------- Inferrable information
    // The kind of geometry
    BravaisLatticeType classifier;
    // Number of sites (total)
    int nos;
    // Number of non-vacancy sites (if defects are activated)
    int nos_nonvacant;
    // Number of basis cells total
    int n_cells_total;
    // Positions of all the atoms
    vectorfield positions;
    // Spin magnetic moments of the atoms
    scalarfield mu_s;
    // Atom types of all the atoms: type index 0..n or or vacancy (type < 0)
    intfield atom_types;
    // Pinning
    intfield mask_unpinned;
    vectorfield mask_pinned_cells;
    // Dimensionality of the points
    int dimensionality;
    int dimensionality_basis;
    // Center and Bounds
    Vector3 center, bounds_min, bounds_max;
    // Unit Cell Bounds
    Vector3 cell_bounds_min, cell_bounds_max;

private:
    // Generate the full set of spin positions
    void generatePositions();
    // Apply the Basis_Cell_Composition to this geometry (i.e. set atom types, mu_s etc.)
    void applyCellComposition();
    // Calculate and update the dimensionality of the points in this geometry
    void calculateDimensionality();
    // Calculate and update bounds of the System
    void calculateBounds();
    // Calculate and update unit cell bounds
    void calculateUnitCellBounds();
    // Calculate and update the type lattice
    void calculateGeometryType();

    //
    std::vector<triangle_t> _triangulation;
    std::vector<tetrahedron_t> _tetrahedra;

    // Temporaries to tell wether the triangulation or tetrahedra
    // need to be updated when the corresponding function is called
    int last_update_n_cell_step;
    intfield last_update_n_cells;
    std::array<int, 6> last_update_cell_ranges;
};

// TODO: find better place (?)
std::vector<triangle_t> compute_delaunay_triangulation_2D( const std::vector<vector2_t> & points );

} // namespace Data

#endif