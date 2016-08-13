#pragma once
#ifndef INTERFACE_GEOMETRY_H
#define INTERFACE_GEOMETRY_H
struct State;

// Get Bounds as array (x,y,z)
extern "C" void Geometry_Get_Bounds_(State *state, float * min, float * max);
// Get Center as array (x,y,z)
extern "C" void Geometry_Get_Center(State *state, float * center);

// Get basis vectors ta, tb, tc
extern "C" void Geometry_Get_Basis_Vectors(State *state, float * a, float * b, float * c);
// TODO: Get basis atoms
// extern "C" void Geometry_Get_Basis_Atoms(State *state, float ** atoms);

// Get number of basis cells in the three translation directions
extern "C" void Geometry_Get_N_Cells(State *state, int * na, int * nb, int * nc);
// Get translation vectors ta, tb, tc
extern "C" void Geometry_Get_Translation_Vectors(State *state, float * ta, float * tb, float * tc);

// Find out if the system is a true 2D system
extern "C" bool Geometry_Is_2D(State * state);

// TODO: Geometry_Get_Spin_Positions

#endif