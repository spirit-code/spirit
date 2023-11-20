#pragma once
#ifndef SPIRIT_CORE_HAMILTONIAN_H
#define SPIRIT_CORE_HAMILTONIAN_H
#include "Spirit_Defines.h"

#include "DLL_Define_Export.h"

struct State;

/*
Hamiltonian
====================================================================

```C
#include "Spirit/Hamiltonian.h"
```

This currently only provides an interface to the Heisenberg Hamiltonian.
*/

/*
DMI chirality
--------------------------------------------------------------------

This means that
- Bloch chirality corresponds to DM vectors along bonds
- Neel chirality corresponds to DM vectors orthogonal to bonds

Neel chirality should therefore only be used in 2D systems.
*/

// Bloch chirality
#define SPIRIT_CHIRALITY_BLOCH 1

// Neel chirality
#define SPIRIT_CHIRALITY_NEEL 2

// Bloch chirality, inverted DM vectors
#define SPIRIT_CHIRALITY_BLOCH_INVERSE -1

// Neel chirality, inverted DM vectors
#define SPIRIT_CHIRALITY_NEEL_INVERSE -2

/*
Dipole-Dipole method
--------------------------------------------------------------------
*/

// Do not use dipolar interactions
#define SPIRIT_DDI_METHOD_NONE 0

// Use fast Fourier transform (FFT) convolutions
#define SPIRIT_DDI_METHOD_FFT 1

// Use the fast multipole method (FMM)
#define SPIRIT_DDI_METHOD_FMM 2

// Use a direct summation with a cutoff radius
#define SPIRIT_DDI_METHOD_CUTOFF 3

/*
Hamiltonian classification id (legacy support)
*/

#define SPIRIT_HAMILTONIAN_CLASS_GENERIC 0

#define SPIRIT_HAMILTONIAN_CLASS_GAUSSIAN 1

#define SPIRIT_HAMILTONIAN_CLASS_HEISENBERG 2
/*
Setters
--------------------------------------------------------------------
*/

// Set the boundary conditions along the translation directions [a, b, c]
PREFIX void Hamiltonian_Set_Boundary_Conditions(
    State * state, const bool * periodical, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Set the (homogeneous) external magnetic field [T]
PREFIX void Hamiltonian_Set_Field(
    State * state, scalar magnitude, const scalar * normal, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Set a global uniaxial anisotropy [meV]
PREFIX void Hamiltonian_Set_Anisotropy(
    State * state, scalar magnitude, const scalar * normal, int idx_image = -1, int idx_chain = -1 ) SUFFIX;
//
// Set a global cubic anisotropy [meV]
PREFIX void
Hamiltonian_Set_Cubic_Anisotropy( State * state, scalar magnitude, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Set the exchange interaction in terms of neighbour shells [meV]
PREFIX void Hamiltonian_Set_Exchange(
    State * state, int n_shells, const scalar * jij, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Set the Dzyaloshinskii-Moriya interaction in terms of neighbour shells [meV]
PREFIX void Hamiltonian_Set_DMI(
    State * state, int n_shells, const scalar * dij, int chirality = SPIRIT_CHIRALITY_BLOCH, int idx_image = -1,
    int idx_chain = -1 ) SUFFIX;

/*
Configure the dipole-dipole interaction

- `ddi_method`: see integers defined above
- `n_periodic_images`: how many repetition of the spin configuration to
  append along the translation directions [a, b, c], if periodical
  boundary conditions are used
- `cutoff_radius`: the distance at which to stop the direct summation,
  if used
- `pb_zero_padding`: if `True` zero padding is used even for periodical directions
*/
PREFIX void Hamiltonian_Set_DDI(
    State * state, int ddi_method, int n_periodic_images[3], scalar cutoff_radius = 0, bool pb_zero_padding = true,
    int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Getters
--------------------------------------------------------------------
*/

// Returns a string containing the name of the Hamiltonian in use
PREFIX const char * Hamiltonian_Get_Name( State * state, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Retrieves the boundary conditions
PREFIX void
Hamiltonian_Get_Boundary_Conditions( State * state, bool * periodical, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Retrieves the external magnetic field [T]
PREFIX void Hamiltonian_Get_Field(
    State * state, scalar * magnitude, scalar * normal, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Retrieves the uniaxial anisotropy [meV]
PREFIX void Hamiltonian_Get_Anisotropy(
    State * state, scalar * magnitude, scalar * normal, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Retrieves the cubic anisotropy [meV]
PREFIX void
Hamiltonian_Get_Cubic_Anisotropy( State * state, scalar * magnitude, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Retrieves the exchange interaction in terms of neighbour shells.

**Note:** if the interactions were specified as pairs, this function
will retrieve `n_shells=0`.
*/
PREFIX void Hamiltonian_Get_Exchange_Shells(
    State * state, int * n_shells, scalar * jij, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Returns the number of exchange interaction pairs
PREFIX int Hamiltonian_Get_Exchange_N_Pairs( State * state, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

PREFIX void Hamiltonian_Get_Exchange_Pairs(
    State * state, int idx[][2], int translations[][3], scalar * Jij, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Retrieves the Dzyaloshinskii-Moriya interaction in terms of neighbour shells.

**Note:** if the interactions were specified as pairs, this function
will retrieve `n_shells=0`.
*/
PREFIX void Hamiltonian_Get_DMI_Shells(
    State * state, int * n_shells, scalar * dij, int * chirality, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Returns the number of Dzyaloshinskii-Moriya interaction pairs
PREFIX int Hamiltonian_Get_DMI_N_Pairs( State * state, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Retrieves the dipole-dipole interaction configuration.

- `ddi_method`: see integers defined above
- `n_periodic_images`: how many repetitions of the spin configuration to
  append along the translation directions [a, b, c], if periodical boundary conditions are used
- `cutoff_radius`: the distance at which to stop the direct summation, if method_cutoff is used
- `pb_zero_padding`: if `True` zero padding is used even for periodical directions
*/
PREFIX void Hamiltonian_Get_DDI(
    State * state, int * ddi_method, int n_periodic_images[3], scalar * cutoff_radius, bool * pb_zero_padding,
    int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Writes the 3Nx3N embedding Hessian to a file.
If triplet_format is set to true the hessian is written as a list of triplets, recommended for large and sparse Hessians.
*/
PREFIX void Hamiltonian_Write_Hessian(
    State * state, const char * filename, bool triplet_format = true, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

#include "DLL_Undefine_Export.h"
#endif
