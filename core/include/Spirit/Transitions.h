#pragma once
#ifndef INTERFACE_TRANSITIONS_H
#define INTERFACE_TRANSITIONS_H
#include "DLL_Define_Export.h"

struct State;

DLLEXPORT void Transition_Homogeneous(State *state, int idx_1, int idx_2, int idx_chain=-1) noexcept;
DLLEXPORT void Transition_Add_Noise_Temperature(State *state, float temperature, int idx_1, int idx_2, int idx_chain=-1) noexcept;

#include "DLL_Undefine_Export.h"
#endif