#pragma once
#ifndef INTERFACE_CONFIGURATIONS_H
#define INTERFACE_CONFIGURATIONS_H
#include "DLL_Define_Export.h"
struct State;


float const defaultPos[3] = {0,0,0}; 
float const defaultRect[3] = {-1,-1,-1};


// Copies the current spin configuration to the clipboard
DLLEXPORT void Configuration_To_Clipboard(State *state, int idx_image=-1, int idx_chain=-1) noexcept;
// Pastes the clipboard spin configuration
DLLEXPORT void Configuration_From_Clipboard(State *state, const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false, int idx_image=-1, int idx_chain=-1) noexcept;
// Pastes the clipboard spin configuration
DLLEXPORT bool Configuration_From_Clipboard_Shift(State *state, const float shift[3], const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted = false, int idx_image=-1, int idx_chain=-1) noexcept;

// Creates a homogeneous domain
DLLEXPORT void Configuration_Domain(State *state, const float direction[3], const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false, int idx_image=-1, int idx_chain=-1) noexcept;
// Points all Spins in +z direction
DLLEXPORT void Configuration_PlusZ(State *state, const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false, int idx_image=-1, int idx_chain=-1) noexcept;
// Points all Spins in -z direction
DLLEXPORT void Configuration_MinusZ(State *state, const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false, int idx_image=-1, int idx_chain=-1) noexcept;

// Points all Spins in random directions
DLLEXPORT void Configuration_Random(State *state, const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false, bool external=false, int idx_image=-1, int idx_chain=-1) noexcept;
// Adds some random noise scaled by temperature
DLLEXPORT void Configuration_Add_Noise_Temperature(State *state, float temperature, const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false, int idx_image=-1, int idx_chain=-1) noexcept;

// Calculate the eigenmodes of the System (Image)
DLLEXPORT void Configuration_Displace_Eigenmode(State *state, int idx_mode, int idx_image=-1, int idx_chain=-1) noexcept;

// Create a toroidal Hopfion
DLLEXPORT void Configuration_Hopfion(State *state, float r, int order=1, const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false,  int idx_image=-1, int idx_chain=-1) noexcept;

// Create a skyrmion configuration
DLLEXPORT void Configuration_Skyrmion(State *state, float r, float order, float phase, bool upDown, bool achiral, bool rl, const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false, int idx_image=-1, int idx_chain=-1) noexcept;

// Spin Spiral
DLLEXPORT void Configuration_SpinSpiral(State *state, const char * direction_type, float q[3], float axis[3], float theta, const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false, int idx_image=-1, int idx_chain=-1) noexcept;
// 2q Spin Spiral
DLLEXPORT void Configuration_SpinSpiral_2q(State *state, const char * direction_type, float q1[3], float q2[3], float axis[3], float theta, const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false, int idx_image=-1, int idx_chain=-1) noexcept;

// Pinning
DLLEXPORT void Configuration_Set_Pinned(State *state, bool pinned, const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false, int idx_image=-1, int idx_chain=-1) noexcept;
// Defects
DLLEXPORT void Configuration_Set_Atom_Type(State *state, int type, const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false, int idx_image=-1, int idx_chain=-1) noexcept;

#include "DLL_Undefine_Export.h"
#endif