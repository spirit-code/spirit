#pragma once
#ifndef INTERFACE_SYSTEM_H
#define INTERFACE_SYSTEM_H
struct State;

// Info
extern "C" int System_Get_Index(State * state);
extern "C" int System_Get_NOS(State * state, int idx_image=-1, int idx_chain=-1);

extern "C" double * System_Get_Spin_Directions(State * state, int idx_image=-1, int idx_chain=-1);

// Data
extern "C" double System_Get_Energy(State * state, int idx_image=-1, int idx_chain=-1);
extern "C" void System_Get_Energy_Array(State * state, double * energies, int idx_image=-1, int idx_chain=-1);

// Update Data (primarily for plots)
extern "C" void System_Update_Data(State * state, int idx_image=-1, int idx_chain=-1);

#endif