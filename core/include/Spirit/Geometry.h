#pragma once
#ifndef INTERFACE_GEOMETRY_H
#define INTERFACE_GEOMETRY_H
#include "DLL_Define_Export.h"
struct State;

#include "Spirit_Defines.h"

// Define Bravais lattice types
typedef enum
{
    Bravais_Lattice_Irregular   = 0,
    Bravais_Lattice_Rectilinear = 1,
    Bravais_Lattice_SC          = 2,
    Bravais_Lattice_Hex2D       = 3,
    Bravais_Lattice_HCP         = 4,
    Bravais_Lattice_BCC         = 5,
    Bravais_Lattice_FCC         = 6
} Bravais_Lattice_Type;

// Set the type of Bravais lattice. Can be e.g. "sc" or "bcc"
DLLEXPORT void Geometry_Set_Bravais_Lattice(State *state, const char * bravais_lattice) noexcept;
// Set the number of basis cells in the three translation directions
DLLEXPORT void Geometry_Set_N_Cells(State * state, int n_cells[3]) noexcept;
// Set the number and positions of atoms in a basis cell
DLLEXPORT void Geometry_Set_Cell_Atoms(State *state, int n_atoms, float ** atoms) noexcept;
// Set the types of the atoms in a basis cell
DLLEXPORT void Geometry_Set_Cell_Atom_Types(State *state, float lattice_constant) noexcept;
// Set the bravais vectors
DLLEXPORT void Geometry_Set_Bravais_Vectors(State *state, float ta[3], float tb[3], float tc[3]) noexcept;
// Set the overall lattice constant
DLLEXPORT void Geometry_Set_Lattice_Constant(State *state, float lattice_constant) noexcept;


// Get number of spins
DLLEXPORT int Geometry_Get_NOS(State * state) noexcept;

// Get positions of spins
DLLEXPORT scalar * Geometry_Get_Positions(State * state, int idx_image=-1, int idx_chain=-1) noexcept;

// Get atom types of lattice sites
DLLEXPORT int * Geometry_Get_Atom_Types(State * state, int idx_image=-1, int idx_chain=-1) noexcept;

// Get Bounds as array (x,y,z)
DLLEXPORT void Geometry_Get_Bounds(State *state, float min[3], float max[3], int idx_image=-1, int idx_chain=-1) noexcept;
// Get Center as array (x,y,z)
DLLEXPORT void Geometry_Get_Center(State *state, float center[3], int idx_image=-1, int idx_chain=-1) noexcept;
// Get Cell Bounds as array (x,y,z)
DLLEXPORT void Geometry_Get_Cell_Bounds(State *state, float min[3], float max[3], int idx_image=-1, int idx_chain=-1) noexcept;

// Get bravais lattice type
DLLEXPORT Bravais_Lattice_Type Geometry_Get_Bravais_Type(State *state, int idx_image=-1, int idx_chain=-1) noexcept;
// Get bravais vectors ta, tb, tc
DLLEXPORT void Geometry_Get_Bravais_Vectors(State *state, float a[3], float b[3], float c[3], int idx_image=-1, int idx_chain=-1) noexcept;
// Get number of atoms in a basis cell
DLLEXPORT int Geometry_Get_N_Cell_Atoms(State *state, int idx_image=-1, int idx_chain=-1) noexcept;
// Get basis cell atoms
DLLEXPORT int Geometry_Get_Cell_Atoms(State *state, scalar ** atoms, int idx_image=-1, int idx_chain=-1);

// Get number of basis cells in the three translation directions
DLLEXPORT void Geometry_Get_N_Cells(State *state, int n_cells[3], int idx_image=-1, int idx_chain=-1) noexcept;
// Get translation vectors ta, tb, tc
DLLEXPORT void Geometry_Get_Translation_Vectors(State *state, float ta[3], float tb[3], float tc[3], int idx_image=-1, int idx_chain=-1) noexcept;

// Retrieve dimensionality of the system (0, 1, 2, 3)
DLLEXPORT int Geometry_Get_Dimensionality(State * state, int idx_image=-1, int idx_chain=-1) noexcept;

// Get the 2D Delaunay triangulation. Returns the number of triangles and
// sets *indices_ptr to point to a list of index 3-tuples.
DLLEXPORT int Geometry_Get_Triangulation(State * state, const int **indices_ptr, int n_cell_step=1, int idx_image=-1, int idx_chain=-1) noexcept;
// Get the 3D Delaunay triangulation. Returns the number of tetrahedrons and
// sets *indices_ptr to point to a list of index 4-tuples.
DLLEXPORT int Geometry_Get_Tetrahedra(State * state, const int **indices_ptr, int n_cell_step=1, int idx_image=-1, int idx_chain=-1) noexcept;


#include "DLL_Undefine_Export.h"
#endif
