#pragma once
#ifndef INTERFACE_CONFIGURATIONS_H
#define INTERFACE_CONFIGURATIONS_H
#include "DLL_Define_Export.h"
struct State;


float const defaultPos[3] = {0,0,0}; 
float const defaultRect[3] = {-1,-1,-1};


// Orients all spins with x>pos into the direction of the v
// DLLEXPORT void Configuration_DomainWall(State *state, const float pos[3], float v[3], const bool greater = true, int idx_image=-1, int idx_chain=-1);

// Creates a homogeneous domain
DLLEXPORT void Configuration_Domain(State *state, const float direction[3], const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false, int idx_image=-1, int idx_chain=-1);
// Points all Spins in +z direction
DLLEXPORT void Configuration_PlusZ(State *state, const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false, int idx_image=-1, int idx_chain=-1);
// Points all Spins in -z direction
DLLEXPORT void Configuration_MinusZ(State *state, const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false, int idx_image=-1, int idx_chain=-1);

// Points all Spins in random directions
DLLEXPORT void Configuration_Random(State *state, const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false, bool external=false, int idx_image=-1, int idx_chain=-1);
// Adds some random noise scaled by temperature
DLLEXPORT void Configuration_Add_Noise_Temperature(State *state, float temperature, const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false, int idx_image=-1, int idx_chain=-1);

// Create a toroidal Hopfion
DLLEXPORT void Configuration_Hopfion(State *state, float r, int order=1, const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false,  int idx_image = -1, int idx_chain = -1);

// Create a skyrmion configuration
DLLEXPORT void Configuration_Skyrmion(State *state, float r, float order, float phase, bool upDown, bool achiral, bool rl, const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false, int idx_image=-1, int idx_chain=-1);

// Spin Spiral
DLLEXPORT void Configuration_SpinSpiral(State *state, const char * direction_type, float q[3], float axis[3], float theta, const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false, int idx_image=-1, int idx_chain=-1);

// TODO: file read
//DLLEXPORT void Configuration_from_File(State * state, const char * filename, int idx_image=-1, int idx_chain=-1);

#include "DLL_Undefine_Export.h"
#endif