#pragma once
#ifndef INTERFACE_CONFIGURATIONS_H
#define INTERFACE_CONFIGURATIONS_H
#include "DLL_Define_Export.h"
struct State;

/*
Configurations
====================================================================

```C
#include "Spirit/Configurations.h"
```

Setting spin configurations for individual spin systems.

The position of the relative center and a set of conditions can be defined.
*/

// The default center position: the center of the system
float const defaultPos[3] = {0,0,0}; 

/*
The default rectangular condition: none

`-1` means that no condition is applied
*/
float const defaultRect[3] = {-1,-1,-1};

/*
Clipboard
--------------------------------------------------------------------
*/

// Copies the current spin configuration to the clipboard
PREFIX void Configuration_To_Clipboard(State *state, int idx_image=-1, int idx_chain=-1) SUFFIX;

// Pastes the clipboard spin configuration
PREFIX void Configuration_From_Clipboard(State *state, const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false, int idx_image=-1, int idx_chain=-1) SUFFIX;

// Pastes the clipboard spin configuration
PREFIX bool Configuration_From_Clipboard_Shift(State *state, const float shift[3], const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted = false, int idx_image=-1, int idx_chain=-1) SUFFIX;

/*
Nonlocalised
--------------------------------------------------------------------
*/

// Creates a homogeneous domain
PREFIX void Configuration_Domain(State *state, const float direction[3], const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false, int idx_image=-1, int idx_chain=-1) SUFFIX;

// Points all spins in +z direction
PREFIX void Configuration_PlusZ(State *state, const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false, int idx_image=-1, int idx_chain=-1) SUFFIX;

// Points all spins in -z direction
PREFIX void Configuration_MinusZ(State *state, const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false, int idx_image=-1, int idx_chain=-1) SUFFIX;

// Points all spins in random directions
PREFIX void Configuration_Random(State *state, const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false, bool external=false, int idx_image=-1, int idx_chain=-1) SUFFIX;

// Spin spiral
PREFIX void Configuration_SpinSpiral(State *state, const char * direction_type, float q[3], float axis[3], float theta, const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false, int idx_image=-1, int idx_chain=-1) SUFFIX;

// 2q spin spiral
PREFIX void Configuration_SpinSpiral_2q(State *state, const char * direction_type, float q1[3], float q2[3], float axis[3], float theta, const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false, int idx_image=-1, int idx_chain=-1) SUFFIX;


/*
Perturbations
--------------------------------------------------------------------
*/

// Adds some random noise scaled by temperature
PREFIX void Configuration_Add_Noise_Temperature(State *state, float temperature, const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false, int idx_image=-1, int idx_chain=-1) SUFFIX;

// Calculate the eigenmodes of the system (Image)
PREFIX void Configuration_Displace_Eigenmode(State *state, int idx_mode, int idx_image=-1, int idx_chain=-1) SUFFIX;

/*
Localised
--------------------------------------------------------------------
*/

// Create a skyrmion configuration
PREFIX void Configuration_Skyrmion(State *state, float r, float order, float phase, bool upDown, bool achiral, bool rl, const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false, int idx_image=-1, int idx_chain=-1) SUFFIX;

// Create a toroidal Hopfion
PREFIX void Configuration_Hopfion(State *state, float r, int order=1, const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false,  int idx_image=-1, int idx_chain=-1) SUFFIX;

/*
Pinning and atom types
--------------------------------------------------------------------

This API can also be used to change the `pinned` state and the `atom type`
of atoms in a spacial region, instead of using translation indices.
*/

// Pinning
PREFIX void Configuration_Set_Pinned(State *state, bool pinned, const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false, int idx_image=-1, int idx_chain=-1) SUFFIX;

// Atom types
PREFIX void Configuration_Set_Atom_Type(State *state, int type, const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false, int idx_image=-1, int idx_chain=-1) SUFFIX;

#include "DLL_Undefine_Export.h"
#endif