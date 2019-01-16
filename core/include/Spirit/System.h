#pragma once
#ifndef INTERFACE_SYSTEM_H
#define INTERFACE_SYSTEM_H
#include "DLL_Define_Export.h"
struct State;

#include "Spirit_Defines.h"

/*
System
====================================================================

```C
#include "Spirit/Transitions.h"
```

Spin systems are often referred to as "images" throughout Spirit.
The `idx_image` is used throughout the API to specify which system
out of the chain a function should be applied to.
`idx_image=-1` refers to the active image of the chain.
*/

// Returns the index of the currently active spin system in the chain.
PREFIX int System_Get_Index(State * state) SUFFIX;

// Returns the number of spins (NOS) of a spin system.
PREFIX int System_Get_NOS(State * state, int idx_image=-1, int idx_chain=-1) SUFFIX;

/*
Returns a pointer to the spin orientations data.

The array is contiguous and of shape (NOS, 3).
*/
PREFIX scalar * System_Get_Spin_Directions(State * state, int idx_image=-1, int idx_chain=-1) SUFFIX;

/*
Returns a pointer to the effective field data.

The array is contiguous and of shape (NOS, 3).
*/
PREFIX scalar * System_Get_Effective_Field(State * state, int idx_image=-1, int idx_chain=-1) SUFFIX;

/*
Returns a pointer to the data of the N'th eigenmode of a spin system.

The array is contiguous and of shape (NOS, 3).
*/
PREFIX scalar * System_Get_Eigenmode(State * state, int idx_mode, int idx_image=-1, int idx_chain=-1) SUFFIX;

// Returns the reaction coordinate of a system along the chain.
PREFIX float System_Get_Rx(State * state, int idx_image=-1, int idx_chain=-1) SUFFIX;

// Returns the energy of a spin system.
PREFIX float System_Get_Energy(State * state, int idx_image=-1, int idx_chain=-1) SUFFIX;

// Retrieves the energy contributions of a spin system.
PREFIX void System_Get_Energy_Array(State * state, float * energies, int idx_image=-1, int idx_chain=-1) SUFFIX;

// Retrieves the eigenvalues of a spin system
PREFIX void System_Get_Eigenvalues(State * state, float * eigenvalues, int idx_image=-1, int idx_chain=-1) SUFFIX;

// Write the energy as formatted output to the console
PREFIX void System_Print_Energy_Array(State * state, int idx_image=-1, int idx_chain=-1) SUFFIX;

// Update Data (primarily for plots)
PREFIX void System_Update_Data(State * state, int idx_image=-1, int idx_chain=-1) SUFFIX;

// Update Eigenmodes (primarily for visualisation or saving)
PREFIX void System_Update_Eigenmodes(State *state, int idx_image=-1, int idx_chain=-1) SUFFIX;

#include "DLL_Undefine_Export.h"
#endif