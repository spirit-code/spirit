#pragma once
#ifndef INTERFACE_QUANTITIES_H
#define INTERFACE_QUANTITIES_H
#include "DLL_Define_Export.h"

struct State;

// Total Magnetization
DLLEXPORT void Quantity_Get_Magnetization(State * state, float m[3], int idx_image=-1, int idx_chain=-1);

// Topological Charge
DLLEXPORT float Quantity_Get_Topological_Charge(State * state, int idx_image=-1, int idx_chain=-1);

#endif