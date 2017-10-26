#pragma once
#ifndef INTERFACE_COLLECTION_H
#define INTERFACE_COLLECTION_H
#include "DLL_Define_Export.h"
struct State;

// Info
DLLEXPORT int Collection_Get_NOC(State * state) noexcept;

// Move
DLLEXPORT void Collection_next_Chain(State * state) noexcept;
DLLEXPORT void Collection_prev_Chain(State * state) noexcept;

#include "DLL_Undefine_Export.h"
#endif