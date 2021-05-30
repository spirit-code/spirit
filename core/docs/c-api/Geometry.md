

Geometry
====================================================================

```C
#include "Spirit/Geometry.h"
```

This set of functions can be used to get information about the
geometric setup of the system and to change it.

Note that it is not fully safe to change the geometry during a
calculation, as this has not been so thoroughly tested.



Definition of Bravais lattice types
--------------------------------------------------------------------

```C
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
```



Setters
--------------------------------------------------------------------



### Geometry_Set_Bravais_Lattice_Type

```C
void Geometry_Set_Bravais_Lattice_Type(State *state, Bravais_Lattice_Type lattice_type)
```

Set the type of Bravais lattice. Can be e.g. "sc" or "bcc".



### Geometry_Set_N_Cells

```C
void Geometry_Set_N_Cells(State * state, int n_cells[3])
```

Set the number of basis cells in the three translation directions.



### Geometry_Set_Cell_Atoms

```C
void Geometry_Set_Cell_Atoms(State *state, int n_atoms, float ** atoms)
```

Set the number and positions of atoms in a basis cell.
Positions are in units of the bravais vectors (scaled by the lattice constant).



### Geometry_Set_mu_s

```C
void Geometry_Set_mu_s(State *state, float mu_s, int idx_image=-1, int idx_chain=-1)
```

Set the magnetic moments of basis cell atoms.



### Geometry_Set_Cell_Atom_Types

```C
void Geometry_Set_Cell_Atom_Types(State *state, int n_atoms, int * atom_types)
```

Set the types of the atoms in a basis cell.



### Geometry_Set_Bravais_Vectors

```C
void Geometry_Set_Bravais_Vectors(State *state, float ta[3], float tb[3], float tc[3])
```

Manually set the bravais vectors.



### Geometry_Set_Lattice_Constant

```C
void Geometry_Set_Lattice_Constant(State *state, float lattice_constant)
```

Set the overall lattice scaling constant.



Getters
--------------------------------------------------------------------



### Geometry_Get_NOS

```C
int Geometry_Get_NOS(State * state)
```

**Returns:** the number of spins.



### Geometry_Get_Positions

```C
scalar * Geometry_Get_Positions(State * state, int idx_image=-1, int idx_chain=-1)
```

**Returns:** pointer to positions of spins (array of length 3*NOS).



### Geometry_Get_Atom_Types

```C
int * Geometry_Get_Atom_Types(State * state, int idx_image=-1, int idx_chain=-1)
```

**Returns:** pointer to atom types of lattice sites.



### Geometry_Get_Bounds

```C
void Geometry_Get_Bounds(State *state, float min[3], float max[3], int idx_image=-1, int idx_chain=-1)
```

Get Bounds as array (x,y,z).



### Geometry_Get_Center

```C
void Geometry_Get_Center(State *state, float center[3], int idx_image=-1, int idx_chain=-1)
```

Get Center as array (x,y,z).



### Geometry_Get_Bravais_Lattice_Type

```C
Bravais_Lattice_Type Geometry_Get_Bravais_Lattice_Type(State *state, int idx_image=-1, int idx_chain=-1)
```

Get bravais lattice type (see the `enum` defined above).



### Geometry_Get_Bravais_Vectors

```C
void Geometry_Get_Bravais_Vectors(State *state, float a[3], float b[3], float c[3], int idx_image=-1, int idx_chain=-1)
```

Get bravais vectors ta, tb, tc.



### Geometry_Get_Dimensionality

```C
int Geometry_Get_Dimensionality(State * state, int idx_image=-1, int idx_chain=-1)
```

Retrieve dimensionality of the system (0, 1, 2, 3).



### Geometry_Get_mu_s

```C
void Geometry_Get_mu_s(State *state, float * mu_s, int idx_image=-1, int idx_chain=-1)
```

Get the magnetic moments of basis cell atoms.



### Geometry_Get_N_Cells

```C
void Geometry_Get_N_Cells(State *state, int n_cells[3], int idx_image=-1, int idx_chain=-1)
```

Get number of basis cells in the three translation directions.



### The basis cell



### Geometry_Get_Cell_Bounds

```C
void Geometry_Get_Cell_Bounds(State *state, float min[3], float max[3], int idx_image=-1, int idx_chain=-1)
```

Get Cell Bounds as array (x,y,z).



### Geometry_Get_N_Cell_Atoms

```C
int Geometry_Get_N_Cell_Atoms(State *state, int idx_image=-1, int idx_chain=-1)
```

Get number of atoms in a basis cell.



### Geometry_Get_Cell_Atoms

```C
int Geometry_Get_Cell_Atoms(State *state, scalar ** atoms, int idx_image=-1, int idx_chain=-1)
```

Get basis cell atom positions in units of the bravais vectors (scaled by the lattice constant).



### Triangulation and tetrahedra



### Geometry_Get_Triangulation

```C
int Geometry_Get_Triangulation(State * state, const int **indices_ptr, int n_cell_step=1, int idx_image=-1, int idx_chain=-1)
```

Get the 2D Delaunay triangulation. Returns the number of triangles and
sets *indices_ptr to point to a list of index 3-tuples.

**Returns:** the number of triangles in the triangulation



### Geometry_Get_Tetrahedra

```C
int Geometry_Get_Tetrahedra(State * state, const int **indices_ptr, int n_cell_step=1, int idx_image=-1, int idx_chain=-1)
```

Get the 3D Delaunay triangulation. Returns the number of tetrahedrons and
sets *indices_ptr to point to a list of index 4-tuples.

**Returns:** the number of tetrahedra

