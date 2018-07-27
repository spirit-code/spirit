#pragma once
#ifndef INTERFACE_PARAMETERS_LLG_H
#define INTERFACE_PARAMETERS_LLG_H
#include "IO.h"
#include "DLL_Define_Export.h"

struct State;

//      Set LLG
// Output
DLLEXPORT void Parameters_LLG_Set_Output_Tag(State *state, const char * tag, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_LLG_Set_Output_Folder(State *state, const char * folder, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_LLG_Set_Output_General(State *state, bool any, bool initial, bool final, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_LLG_Set_Output_Energy(State *state, bool energy_step, bool energy_archive, bool energy_spin_resolved, bool energy_divide_by_nos, bool energy_add_readability_lines, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_LLG_Set_Output_Configuration(State *state, bool configuration_step, bool configuration_archive, int configuration_filetype=IO_Fileformat_OVF_text, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_LLG_Set_N_Iterations(State *state, int n_iterations, int n_iterations_log, int idx_image=-1, int idx_chain=-1) noexcept;
// Simulation Parameters
DLLEXPORT void Parameters_LLG_Set_Direct_Minimization(State *state, bool direct, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_LLG_Set_Convergence(State *state, float convergence, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_LLG_Set_Time_Step(State *state, float dt, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_LLG_Set_Damping(State *state, float damping, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_LLG_Set_STT(State *state, bool use_gradient, float magnitude, const float normal[3], int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_LLG_Set_Temperature(State *state, float T, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_LLG_Set_Temperature_Gradient(State *state, float inclination, const float direction[3], int idx_image=-1, int idx_chain=-1) noexcept;

//      Get LLG
// Output
DLLEXPORT const char * Parameters_LLG_Get_Output_Tag(State *state, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT const char * Parameters_LLG_Get_Output_Folder(State *state, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_LLG_Get_Output_General(State *state, bool * any, bool * initial, bool * final, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_LLG_Get_Output_Energy(State *state, bool * energy_step, bool * energy_archive, bool * energy_spin_resolved, bool * energy_divide_by_nos, bool * energy_add_readability_lines, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_LLG_Get_Output_Configuration(State *state, bool * configuration_step, bool * configuration_archive, int * configuration_filetype, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_LLG_Get_N_Iterations(State *state, int * iterations, int * iterations_log, int idx_image=-1, int idx_chain=-1) noexcept;
// Simulation Parameters
DLLEXPORT bool Parameters_LLG_Get_Direct_Minimization(State *state, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT float Parameters_LLG_Get_Convergence(State *state, int idx_image=-1, int idx_chain=-1) noexcept;
// Set the LLG time step in [ps]
DLLEXPORT float Parameters_LLG_Get_Time_Step(State *state, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT float Parameters_LLG_Get_Damping(State *state, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT float Parameters_LLG_Get_Temperature(State *state, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_LLG_Get_Temperature_Gradient(State *state, float * direction, float normal[3], int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_LLG_Get_STT(State *state, bool * use_gradient, float * magnitude, float normal[3], int idx_image=-1, int idx_chain=-1) noexcept;

#include "DLL_Undefine_Export.h"
#endif