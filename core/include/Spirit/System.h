#pragma once
#ifndef INTERFACE_SYSTEM_H
#define INTERFACE_SYSTEM_H
#include "DLL_Define_Export.h"

struct State;

#include "Spirit_Defines.h"

// Info
PREFIX int System_Get_Index(State * state) SUFFIX;
PREFIX int System_Get_NOS(State * state, int idx_image=-1, int idx_chain=-1) SUFFIX;

// Data
PREFIX scalar * System_Get_Spin_Directions(State * state, int idx_image=-1, int idx_chain=-1) SUFFIX;
PREFIX scalar * System_Get_Effective_Field(State * state, int idx_image=-1, int idx_chain=-1) SUFFIX;
PREFIX scalar * System_Get_Eigenmode(State * state, int idx_mode, int idx_image=-1, int idx_chain=-1) SUFFIX;
PREFIX float System_Get_Rx(State * state, int idx_image=-1, int idx_chain=-1) SUFFIX;
PREFIX float System_Get_Energy(State * state, int idx_image=-1, int idx_chain=-1) SUFFIX;
PREFIX void System_Get_Energy_Array(State * state, float * energies, int idx_image=-1, int idx_chain=-1) SUFFIX;
PREFIX void System_Get_Eigenvalues(State * state, float * eigenvalues, int idx_image=-1, int idx_chain=-1) SUFFIX;

// Console Output
PREFIX void System_Print_Energy_Array(State * state, int idx_image=-1, int idx_chain=-1) SUFFIX;

// Update Data (primarily for plots)
PREFIX void System_Update_Data(State * state, int idx_image=-1, int idx_chain=-1) SUFFIX;

// Update Eigenmodes (primarily for visualisation or saving)
PREFIX void System_Update_Eigenmodes(State *state, int idx_image=-1, int idx_chain=-1) SUFFIX;

#include "DLL_Undefine_Export.h"
#endif