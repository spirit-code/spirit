#pragma once
#ifndef INTERFACE_PARAMETERS_H
#define INTERFACE_PARAMETERS_H
#include "DLL_Define_Export.h"

struct State;

// Set LLG
DLLEXPORT void Parameters_Set_LLG_Time_Step(State *state, float dt, int idx_image=-1, int idx_chain=-1);
DLLEXPORT void Parameters_Set_LLG_Damping(State *state, float damping, int idx_image=-1, int idx_chain=-1);
DLLEXPORT void Parameters_Set_LLG_N_Iterations(State *state, int n_iterations, int idx_image=-1, int idx_chain=-1);
DLLEXPORT void Parameters_Set_LLG_N_Iterations_Log(State *state, int n_iterations, int idx_image=-1, int idx_chain=-1);
// Set GNEB
DLLEXPORT void Parameters_Set_GNEB_Spring_Constant(State *state, float spring_constant, int idx_image=-1, int idx_chain=-1);
DLLEXPORT void Parameters_Set_GNEB_Climbing_Falling(State *state, bool climbing, bool falling, int idx_image=-1, int idx_chain=-1);
DLLEXPORT void Parameters_Set_GNEB_N_Iterations(State *state, int n_iterations, int idx_image=-1, int idx_chain=-1);
DLLEXPORT void Parameters_Set_GNEB_N_Iterations_Log(State *state, int n_iterations_log, int idx_image=-1, int idx_chain=-1);


// Get LLG
DLLEXPORT void Parameters_Get_LLG_Time_Step(State *state, float * dt, int idx_image=-1, int idx_chain=-1);
DLLEXPORT void Parameters_Get_LLG_Damping(State *state, float * damping, int idx_image=-1, int idx_chain=-1);
DLLEXPORT int Parameters_Get_LLG_N_Iterations(State *state, int idx_image=-1, int idx_chain=-1);
DLLEXPORT int Parameters_Get_LLG_N_Iterations_Log(State *state, int idx_image=-1, int idx_chain=-1);
// Set GNEB
DLLEXPORT void Parameters_Get_GNEB_Spring_Constant(State *state, float * spring_constant, int idx_image=-1, int idx_chain=-1);
DLLEXPORT void Parameters_Get_GNEB_Climbing_Falling(State *state, bool * climbing, bool * falling, int idx_image=-1, int idx_chain=-1);
DLLEXPORT int Parameters_Get_GNEB_N_Iterations(State *state, int idx_image=-1, int idx_chain=-1);
DLLEXPORT int Parameters_Get_GNEB_N_Iterations_Log(State *state, int idx_image=-1, int idx_chain=-1);

#include "DLL_Undefine_Export.h"
#endif