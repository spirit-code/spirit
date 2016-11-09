#pragma once
#ifndef INTERFACE_CONFIGURATIONS_H
#define INTERFACE_CONFIGURATIONS_H
#include "DLL_Define_Export.h"
struct State;

// Orients all spins with x>pos into the direction of the v
DLLEXPORT void Configuration_DomainWall(State *state, const float pos[3], float v[3], const bool greater = true, int idx_image=-1, int idx_chain=-1);

// Points all Spins parallel to the direction of v
// Calls DomainWall (s, -1E+20, v)
DLLEXPORT void Configuration_Homogeneous(State *state, float v[3], int idx_image=-1, int idx_chain=-1);
// Points all Spins in +z direction
DLLEXPORT void Configuration_PlusZ(State *state, int idx_image=-1, int idx_chain=-1);
// Points all Spins in -z direction
DLLEXPORT void Configuration_MinusZ(State *state, int idx_image=-1, int idx_chain=-1);

// Points all Spins in random directions
DLLEXPORT void Configuration_Random(State *state, bool external = false, int idx_image=-1, int idx_chain=-1);
// Adds some random noise scaled by temperature
DLLEXPORT void Configuration_Add_Noise_Temperature(State *state, float temperature, int idx_image=-1, int idx_chain=-1);

// Create a toroidal Hopfion
DLLEXPORT void Configuration_Hopfion(State *state, float pos[3], float r, int idx_image = -1, int idx_chain = -1);

// Points a sperical region of spins of radius r
// into direction of vec at position pos
DLLEXPORT void Configuration_Skyrmion(State *state, float pos[3], float r, float order, float phase, bool upDown, bool achiral, bool rl, int idx_image=-1, int idx_chain=-1);
// Spin Spiral
DLLEXPORT void Configuration_SpinSpiral(State *state, const char * direction_type, float q[3], float axis[3], float theta, int idx_image=-1, int idx_chain=-1);

// TODO: file read
//DLLEXPORT void Configuration_from_File(State * state, const char * filename, int idx_image=-1, int idx_chain=-1);

#include "DLL_Undefine_Export.h"
#endif