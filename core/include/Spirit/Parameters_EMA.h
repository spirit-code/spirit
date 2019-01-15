#pragma once
#ifndef INTERFACE_PARAMETERS_EMA_H
#define INTERFACE_PARAMETERS_EMA_H
#include "IO.h"
#include "DLL_Define_Export.h"

struct State;

//      Set EMA
// Simulation Parameters
PREFIX void Parameters_EMA_Set_N_Modes(State *state, int n_modes, int idx_image=-1, int idx_chain=-1) SUFFIX;
PREFIX void Parameters_EMA_Set_N_Mode_Follow(State *state, int n_mode_follow, int idx_image=-1, int idx_chain=-1) SUFFIX;
PREFIX void Parameters_EMA_Set_Frequency(State *state, float frequency, int idx_image=-1, int idx_chain=-1) SUFFIX;
PREFIX void Parameters_EMA_Set_Amplitude(State *state, float amplitude, int idx_image=-1, int idx_chain=-1) SUFFIX;
PREFIX void Parameters_EMA_Set_Snapshot(State *state, bool snapshot, int idx_image=-1, int idx_chain=-1) SUFFIX;

//      Get EMA
// Simulation Parameters
PREFIX int Parameters_EMA_Get_N_Modes(State *state, int idx_image=-1, int idx_chain=-1) SUFFIX;
PREFIX int Parameters_EMA_Get_N_Mode_Follow(State *state, int idx_image=-1, int idx_chain=-1) SUFFIX;
PREFIX float Parameters_EMA_Get_Frequency(State *state, int idx_image=-1, int idx_chain=-1) SUFFIX;
PREFIX float Parameters_EMA_Get_Amplitude(State *state, int idx_image=-1, int idx_chain=-1) SUFFIX;
PREFIX bool Parameters_EMA_Get_Snapshot(State *state, int idx_image=-1, int idx_chain=-1) SUFFIX;

#include "DLL_Undefine_Export.h"
#endif