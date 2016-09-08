#pragma once
#ifndef INTERFACE_GEOMETRY_H
#define INTERFACE_GEOMETRY_H
struct State;

// Get Positions of Spins
extern "C" double * Geometry_Get_Spin_Positions(State * state, int idx_image=-1, int idx_chain=-1);

// Get Bounds as array (x,y,z)
extern "C" void Geometry_Get_Bounds(State *state, float * min, float * max, int idx_image=-1, int idx_chain=-1);
// Get Center as array (x,y,z)
extern "C" void Geometry_Get_Center(State *state, float * center, int idx_image=-1, int idx_chain=-1);

// Get basis vectors ta, tb, tc
extern "C" void Geometry_Get_Basis_Vectors(State *state, float * a, float * b, float * c, int idx_image=-1, int idx_chain=-1);
// TODO: Get basis atoms
// extern "C" void Geometry_Get_Basis_Atoms(State *state, float ** atoms);

// Get number of basis cells in the three translation directions
extern "C" void Geometry_Get_N_Cells(State *state, int * n_cells, int idx_image=-1, int idx_chain=-1);
// Get translation vectors ta, tb, tc
extern "C" void Geometry_Get_Translation_Vectors(State *state, float * ta, float * tb, float * tc, int idx_image=-1, int idx_chain=-1);

// Find out if the system is a true 2D system
extern "C" bool Geometry_Is_2D(State * state, int idx_image=-1, int idx_chain=-1);

#endif