#pragma once
#ifndef INTERFACE_PARAMETERS_H
#define INTERFACE_PARAMETERS_H
#include "DLL_Define_Export.h"

struct State;

//      Set LLG
// Output
DLLEXPORT void Parameters_Set_LLG_Output_Folder(State *state, const char * folder, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_Set_LLG_Output_General(State *state, bool any, bool initial, bool final, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_Set_LLG_Output_Energy(State *state, bool energy_step, bool energy_archive, bool energy_spin_resolved, bool energy_divide_by_nos, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_Set_LLG_Output_Configuration(State *state, bool configuration_step, bool configuration_archive, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_Set_LLG_N_Iterations(State *state, int n_iterations, int n_iterations_log, int idx_image=-1, int idx_chain=-1) noexcept;
// Simulation Parameters
DLLEXPORT void Parameters_Set_LLG_Direct_Minimization(State *state, bool direct, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_Set_LLG_Convergence(State *state, float convergence, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_Set_LLG_Time_Step(State *state, float dt, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_Set_LLG_Damping(State *state, float damping, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_Set_LLG_STT(State *state, bool use_gradient, float magnitude, const float normal[3], int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_Set_LLG_Temperature(State *state, float T, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_Set_LLG_Temperature_Gradient(State *state, float inclination, const float direction[3], int idx_image=-1, int idx_chain=-1) noexcept;

//      Set MC
// Output
DLLEXPORT void Parameters_Set_MC_Output_Folder(State *state, const char * folder, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_Set_MC_Output_General(State *state, bool any, bool initial, bool final, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_Set_MC_Output_Energy(State *state, bool energy_step, bool energy_archive, bool energy_spin_resolved, bool energy_divide_by_nos, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_Set_MC_Output_Configuration(State *state, bool configuration_step, bool configuration_archive, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_Set_MC_N_Iterations(State *state, int n_iterations, int n_iterations_log, int idx_image=-1, int idx_chain=-1) noexcept;
// Simulation Parameters
DLLEXPORT void Parameters_Set_MC_Temperature(State *state, float T, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_Set_MC_Acceptance_Ratio(State *state, float ratio, int idx_image=-1, int idx_chain=-1) noexcept;

//      Set GNEB
// Output
DLLEXPORT void Parameters_Set_GNEB_Output_Folder(State *state, const char * folder, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_Set_GNEB_Output_General(State *state, bool any, bool initial, bool final, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_Set_GNEB_Output_Energies(State *state, bool energies_step, bool energies_interpolated, bool energies_divide_by_nos, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_Set_GNEB_Output_Chain(State *state, bool chain_step, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_Set_GNEB_N_Iterations(State *state, int n_iterations, int n_iterations_log, int idx_chain=-1) noexcept;
// Simulation Parameters
DLLEXPORT void Parameters_Set_GNEB_Convergence(State *state, float convergence, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_Set_GNEB_Spring_Constant(State *state, float spring_constant, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_Set_GNEB_Climbing_Falling(State *state, int image_type, int idx_image=-1, int idx_chain=-1) noexcept;
// Automatically set the image type for a chain.
//    Maxima are set to climbing, minima to falling, others are not changed.
DLLEXPORT void Parameters_Set_GNEB_Image_Type_Automatically(State *state, int idx_chain=-1) noexcept;


//      Get LLG
// Output
DLLEXPORT const char * Parameters_Get_LLG_Output_Folder(State *state, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_Get_LLG_Output_General(State *state, bool * any, bool * initial, bool * final, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_Get_LLG_Output_Energy(State *state, bool * energy_step, bool * energy_archive, bool * energy_spin_resolved, bool * energy_divide_by_nos, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_Get_LLG_Output_Configuration(State *state, bool * configuration_step, bool * configuration_archive, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_Get_LLG_N_Iterations(State *state, int * iterations, int * iterations_log, int idx_image=-1, int idx_chain=-1) noexcept;
// Simulation Parameters
DLLEXPORT bool Parameters_Get_LLG_Direct_Minimization(State *state, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT float Parameters_Get_LLG_Convergence(State *state, int idx_image=-1, int idx_chain=-1) noexcept;
// Set the LLG time step in [ps]
DLLEXPORT float Parameters_Get_LLG_Time_Step(State *state, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT float Parameters_Get_LLG_Damping(State *state, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT float Parameters_Get_LLG_Temperature(State *state, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_Get_LLG_Temperature_Gradient(State *state, float * direction, float normal[3], int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_Get_LLG_STT(State *state, bool * use_gradient, float * magnitude, float normal[3], int idx_image=-1, int idx_chain=-1) noexcept;

//      Get MC
// Output
DLLEXPORT const char * Parameters_Get_MC_Output_Folder(State *state, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_Get_MC_Output_General(State *state, bool * any, bool * initial, bool * final, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_Get_MC_Output_Energy(State *state, bool * energy_step, bool * energy_archive, bool * energy_spin_resolved, bool * energy_divide_by_nos, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_Get_MC_Output_Configuration(State *state, bool * configuration_step, bool * configuration_archive, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_Get_MC_N_Iterations(State *state, int * iterations, int * iterations_log, int idx_image=-1, int idx_chain=-1) noexcept;
// Simulation Parameters
DLLEXPORT float Parameters_Get_MC_Temperature(State *state, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT float Parameters_Get_MC_Acceptance_Ratio(State *state, int idx_image=-1, int idx_chain=-1) noexcept;

//      Get GNEB
// Output
DLLEXPORT const char * Parameters_Get_GNEB_Output_Folder(State *state, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_Get_GNEB_Output_General(State *state, bool * any, bool * initial, bool * final, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_Get_GNEB_Output_Energies(State *state, bool * energies_step, bool * energies_interpolated, bool * energies_divide_by_nos, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_Get_GNEB_Output_Chain(State *state, bool * chain_step, int idx_chain=-1) noexcept;
DLLEXPORT void Parameters_Get_GNEB_N_Iterations(State *state, int * iterations, int * iterations_log, int idx_chain=-1) noexcept;
// Simulation Parameters
DLLEXPORT float Parameters_Get_GNEB_Convergence(State *state, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT float Parameters_Get_GNEB_Spring_Constant(State *state, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT int Parameters_Get_GNEB_Climbing_Falling(State *state, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT int Parameters_Get_GNEB_N_Energy_Interpolations(State *state, int idx_chain=-1) noexcept;

#include "DLL_Undefine_Export.h"
#endif