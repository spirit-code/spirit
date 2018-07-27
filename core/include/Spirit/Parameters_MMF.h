#pragma once
#ifndef INTERFACE_PARAMETERS_MMF_H
#define INTERFACE_PARAMETERS_MMF_H
#include "IO.h"
#include "DLL_Define_Export.h"

struct State;

//      Set MMF
// Output
DLLEXPORT void Parameters_MMF_Set_Output_Tag(State *state, const char * tag, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_MMF_Set_Output_Folder(State *state, const char * folder, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_MMF_Set_Output_General(State *state, bool any, bool initial, bool final, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_MMF_Set_Output_Energy(State *state, bool energy_step, bool energy_archive, bool energy_spin_resolved, bool energy_divide_by_nos, bool energy_add_readability_lines, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_MMF_Set_Output_Configuration(State *state, bool configuration_step, bool configuration_archive, int configuration_filetype, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_MMF_Set_N_Iterations(State *state, int n_iterations, int n_iterations_log, int idx_image=-1, int idx_chain=-1) noexcept;
// Simulation Parameters
DLLEXPORT void Parameters_MMF_Set_N_Modes(State *state, int n_modes, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_MMF_Set_N_Mode_Follow(State *state, int n_mode_follow, int idx_image=-1, int idx_chain=-1) noexcept;

//      Get MMF
// Output
DLLEXPORT const char * Parameters_MMF_Get_Output_Tag(State *state, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT const char * Parameters_MMF_Get_Output_Folder(State *state, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_MMF_Get_Output_General(State *state, bool * any, bool * initial, bool * final, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_MMF_Get_Output_Energy(State *state, bool * energy_step, bool * energy_archive, bool * energy_spin_resolved, bool * energy_divide_by_nos, bool * energy_add_readability_lines, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_MMF_Get_Output_Configuration(State *state, bool * configuration_step, bool * configuration_archive, int * configuration_filetype, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_MMF_Get_N_Iterations(State *state, int * iterations, int * iterations_log, int idx_image=-1, int idx_chain=-1) noexcept;
// Simulation Parameters
DLLEXPORT int Parameters_MMF_Get_N_Modes(State *state, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT int Parameters_MMF_Get_N_Mode_Follow(State *state, int idx_image=-1, int idx_chain=-1) noexcept;

#include "DLL_Undefine_Export.h"
#endif