#pragma once
#ifndef INTERFACE_SIMULATION_H
#define INTERFACE_SIMULATION_H
#include "DLL_Define_Export.h"

struct State;

#include <vector>

// Single Optimization iteration with a Method
//		To be used with caution! It does not inquire if an iteration is allowed!
DLLEXPORT void Simulation_SingleShot(State *state, const char * c_method_type, const char * c_optimizer_type, 
	int n_iterations = -1, int n_iterations_log = -1, int idx_image=-1, int idx_chain=-1);

// Play/Pause functionality
DLLEXPORT void Simulation_PlayPause(State *state, const char * c_method_type, const char * c_optimizer_type, 
	int n_iterations = -1, int n_iterations_log = -1, int idx_image=-1, int idx_chain=-1);

// Stop all simulations
DLLEXPORT void Simulation_Stop_All(State *state);


// Get maximum Torque component
//		If an LLG simulation is running this returns the max. torque on the current image.
//		If a GNEB simulation is running this returns the max. torque on the current chain.
//		IF a MMF simulation is running this returns the max. torque on the current collection.
DLLEXPORT float Simulation_Get_MaxTorqueComponent(State * state, int idx_image = -1, int idx_chain = -1);

// Get IPS
//		If an LLG simulation is running this returns the IPS on the current image.
//		If a GNEB simulation is running this returns the IPS on the current chain.
//		IF a MMF simulation is running this returns the IPS on the current collection.
DLLEXPORT float Simulation_Get_IterationsPerSecond(State *state, int idx_image = -1, int idx_chain = -1);


// Get IPS
//		If an LLG simulation is running this returns the Optimizer name on the current image.
//		If a GNEB simulation is running this returns the Optimizer name on the current chain.
//		IF a MMF simulation is running this returns the Optimizer name on the current collection.
DLLEXPORT const char * Simulation_Get_Optimizer_Name(State *state, int idx_image = -1, int idx_chain = -1);

// Get IPS
//		If an LLG simulation is running this returns the Method name on the current image.
//		If a GNEB simulation is running this returns the Method name on the current chain.
//		IF a MMF simulation is running this returns the Method name on the current collection.
DLLEXPORT const char * Simulation_Get_Method_Name(State *state, int idx_image = -1, int idx_chain = -1);


// Check for running simulations
DLLEXPORT bool Simulation_Running_Any_Anywhere(State *state);
DLLEXPORT bool Simulation_Running_LLG_Anywhere(State *state);
DLLEXPORT bool Simulation_Running_GNEB_Anywhere(State *state);

DLLEXPORT bool Simulation_Running_LLG_Chain(State *state, int idx_chain=-1);

DLLEXPORT bool Simulation_Running_Any(State *state, int idx_image=-1, int idx_chain=-1);
DLLEXPORT bool Simulation_Running_LLG(State *state, int idx_image=-1, int idx_chain=-1);
DLLEXPORT bool Simulation_Running_GNEB(State *state, int idx_chain=-1);
DLLEXPORT bool Simulation_Running_MMF(State *state);

#include "DLL_Undefine_Export.h"
#endif