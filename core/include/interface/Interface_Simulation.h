#pragma once
#ifndef INTERFACE_SIMULATION_H
#define INTERFACE_SIMULATION_H
struct State;

#include <vector>

// Single Optimization iteration with a Method
//		To be used with caution! It does not inquire if an iteration is allowed!
extern "C" void Simulation_SingleShot(State *state, const char * c_method_type, const char * c_optimizer_type, 
	int n_iterations = -1, int n_iterations_log = -1, int idx_image=-1, int idx_chain=-1);

// Play/Pause functionality
extern "C" void Simulation_PlayPause(State *state, const char * c_method_type, const char * c_optimizer_type, 
	int n_iterations = -1, int n_iterations_log = -1, int idx_image=-1, int idx_chain=-1);

// Stop all simulations
extern "C" void Simulation_Stop_All(State *state);

// Get IPS
std::vector<double> Simulation_Get_IterationsPerSecond(State *state);

// Check for running simulations
extern "C" bool Simulation_Running_Any_Anywhere(State *state);
extern "C" bool Simulation_Running_LLG_Anywhere(State *state);
extern "C" bool Simulation_Running_GNEB_Anywhere(State *state);

extern "C" bool Simulation_Running_LLG_Chain(State *state, int idx_chain=-1);

extern "C" bool Simulation_Running_Any(State *state, int idx_image=-1, int idx_chain=-1);
extern "C" bool Simulation_Running_LLG(State *state, int idx_image=-1, int idx_chain=-1);
extern "C" bool Simulation_Running_GNEB(State *state, int idx_chain=-1);
extern "C" bool Simulation_Running_MMF(State *state);

#endif