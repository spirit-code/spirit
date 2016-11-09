#pragma once
#ifndef INTERFACE_SYSTEM_H
#define INTERFACE_SYSTEM_H
#include "DLL_Define_Export.h"

struct State;

// Info
DLLEXPORT int System_Get_Index(State * state);
DLLEXPORT int System_Get_NOS(State * state, int idx_image=-1, int idx_chain=-1);

// Data
DLLEXPORT double * System_Get_Spin_Directions(State * state, int idx_image=-1, int idx_chain=-1);
DLLEXPORT double * System_Get_Effective_Field(State * state, int idx_image=-1, int idx_chain=-1);
DLLEXPORT double System_Get_Rx(State * state, int idx_image=-1, int idx_chain=-1);
DLLEXPORT double System_Get_Energy(State * state, int idx_image=-1, int idx_chain=-1);
DLLEXPORT void System_Get_Energy_Array(State * state, double * energies, int idx_image=-1, int idx_chain=-1);

// Update Data (primarily for plots)
DLLEXPORT void System_Update_Data(State * state, int idx_image=-1, int idx_chain=-1);

#include "DLL_Undefine_Export.h"
#endif