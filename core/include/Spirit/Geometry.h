#pragma once
#ifndef SPIRIT_CORE_GEOMETRY_H
#define SPIRIT_CORE_GEOMETRY_H
#include "DLL_Define_Export.h"

#include "Spirit_Defines.h"

struct State;

/*
Geometry
====================================================================

```C
#include "Spirit/Geometry.h"
```

This set of functions can be used to get information about the
geometric setup of the system and to change it.

Note that it is not fully safe to change the geometry during a
calculation, as this has not been so thoroughly tested.
*/

/*
Definition of Bravais lattice types
--------------------------------------------------------------------
*/

typedef enum
{
    Bravais_Lattice_Irregular   = 0,
    Bravais_Lattice_Rectilinear = 1,
    Bravais_Lattice_SC          = 2,
    Bravais_Lattice_Hex2D       = 3,
    Bravais_Lattice_Hex2D_60    = 4,
    Bravais_Lattice_Hex2D_120   = 5,
    Bravais_Lattice_HCP         = 6,
    Bravais_Lattice_BCC         = 7,
    Bravais_Lattice_FCC         = 8
} Bravais_Lattice_Type;

/*
Setters
--------------------------------------------------------------------
*/

/*
Set the type of Bravais lattice. Can be e.g. "sc" or "bcc".
*/
PREFIX void Geometry_Set_Bravais_Lattice_Type( State * state, Bravais_Lattice_Type lattice_type ) SUFFIX;

/*
Set the number of basis cells in the three translation directions.
*/
PREFIX void Geometry_Set_N_Cells( State * state, int n_cells[3] ) SUFFIX;

/*
Set the number and positions of atoms in a basis cell.
Positions are in units of the bravais vectors (scaled by the lattice constant).
*/
PREFIX void Geometry_Set_Cell_Atoms( State * state, int n_atoms, float ** atoms ) SUFFIX;

/*
Set the magnetic moments of basis cell atoms.
*/
PREFIX void Geometry_Set_mu_s( State * state, float mu_s, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Set the types of the atoms in a basis cell.
*/
PREFIX void Geometry_Set_Cell_Atom_Types( State * state, int n_atoms, int * atom_types ) SUFFIX;

/*
Manually set the bravais vectors.
*/
PREFIX void Geometry_Set_Bravais_Vectors( State * state, float ta[3], float tb[3], float tc[3] ) SUFFIX;

/*
Set the overall lattice scaling constant.
*/
PREFIX void Geometry_Set_Lattice_Constant( State * state, float lattice_constant ) SUFFIX;

/*
Getters
--------------------------------------------------------------------
*/

/*
**Returns:** the number of spins.
*/
PREFIX int Geometry_Get_NOS( State * state ) SUFFIX;

/*
**Returns:** pointer to positions of spins (array of length 3*NOS).
*/
PREFIX scalar * Geometry_Get_Positions( State * state, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
**Returns:** pointer to atom types of lattice sites.
*/
PREFIX int * Geometry_Get_Atom_Types( State * state, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Get Bounds as array (x,y,z).
*/
PREFIX void
Geometry_Get_Bounds( State * state, float min[3], float max[3], int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Get Center as array (x,y,z).
*/
PREFIX void Geometry_Get_Center( State * state, float center[3], int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Get bravais lattice type (see the `enum` defined above).
*/
PREFIX Bravais_Lattice_Type
Geometry_Get_Bravais_Lattice_Type( State * state, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Get bravais vectors ta, tb, tc.
*/
PREFIX void Geometry_Get_Bravais_Vectors(
    State * state, float a[3], float b[3], float c[3], int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Retrieve dimensionality of the system (0, 1, 2, 3).
*/
PREFIX int Geometry_Get_Dimensionality( State * state, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Get the magnetic moments of basis cell atoms.
*/
PREFIX void Geometry_Get_mu_s( State * state, float * mu_s, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Get number of basis cells in the three translation directions.
*/
PREFIX void Geometry_Get_N_Cells( State * state, int n_cells[3], int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
### The basis cell
*/

/*
Get Cell Bounds as array (x,y,z).
*/
PREFIX void
Geometry_Get_Cell_Bounds( State * state, float min[3], float max[3], int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Get number of atoms in a basis cell.
*/
PREFIX int Geometry_Get_N_Cell_Atoms( State * state, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Get basis cell atom positions in units of the bravais vectors (scaled by the lattice constant).
*/
PREFIX int Geometry_Get_Cell_Atoms( State * state, scalar ** atoms, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
### Triangulation and tetrahedra
*/

/*
Get the 2D Delaunay triangulation. Returns the number of triangles and
sets *indices_ptr to point to a list of index 3-tuples.

**Returns:** the number of triangles in the triangulation
*/
PREFIX int Geometry_Get_Triangulation(
    State * state, const int ** indices_ptr, int n_cell_step = 1, int idx_image = -1, int idx_chain = -1 ) SUFFIX;
PREFIX int Geometry_Get_Triangulation_Ranged(
    State * state, const int ** indices_ptr, int n_cell_step, int ranges[6], int idx_image = -1,
    int idx_chain = -1 ) SUFFIX;

/*
Get the 3D Delaunay triangulation. Returns the number of tetrahedrons and
sets *indices_ptr to point to a list of index 4-tuples.

**Returns:** the number of tetrahedra
*/
PREFIX int Geometry_Get_Tetrahedra(
    State * state, const int ** indices_ptr, int n_cell_step = 1, int idx_image = -1, int idx_chain = -1 ) SUFFIX;
PREFIX int Geometry_Get_Tetrahedra_Ranged(
    State * state, const int ** indices_ptr, int n_cell_step, int ranges[6], int idx_image = -1,
    int idx_chain = -1 ) SUFFIX;

#include "DLL_Undefine_Export.h"
#endif
