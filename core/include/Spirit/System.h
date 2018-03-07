#pragma once
#ifndef INTERFACE_SYSTEM_H
#define INTERFACE_SYSTEM_H
#include "DLL_Define_Export.h"

struct State;

#include "Spirit_Defines.h"

// Info
DLLEXPORT int System_Get_Index(State * state) noexcept;
DLLEXPORT int System_Get_NOS(State * state, int idx_image=-1, int idx_chain=-1) noexcept;

// Data
DLLEXPORT scalar * System_Get_Spin_Directions(State * state, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT scalar * System_Get_Effective_Field(State * state, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT float System_Get_Rx(State * state, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT float System_Get_Energy(State * state, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void System_Get_Energy_Array(State * state, float * energies, int idx_image=-1, int idx_chain=-1) noexcept;

// Console Output
DLLEXPORT void System_Print_Energy_Array(State * state, int idx_image=-1, int idx_chain=-1) noexcept;

// Update Data (primarily for plots)
DLLEXPORT void System_Update_Data(State * state, int idx_image=-1, int idx_chain=-1) noexcept;

// Update Eigenmodes (primarily for visualisation or saving)
DLLEXPORT void System_Update_Eigenmodes(State *state, int idx_image=-1, int idx_chain=-1) noexcept;

#include "DLL_Undefine_Export.h"
#endif