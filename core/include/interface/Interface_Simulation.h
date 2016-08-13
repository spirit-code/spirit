#pragma once
#ifndef INTERFACE_SIMULATION_H
#define INTERFACE_SIMULATION_H
struct State;

#include <vector>

// Play/Pause functionality
extern "C" void Simulation_PlayPause(State *state, const char * c_method_type, const char * c_optimizer_type, 
	int n_iterations = -1, int log_steps = -1, int idx_image=-1, int idx_chain=-1);

// Stop all simulations
extern "C" void Simulation_Stop_All(State *state);

// Get IPS
std::vector<double> Simulation_Get_IterationsPerSecond(State *state);

// Check for running simulations
extern "C" bool Simulation_Running_Any(State *state);
extern "C" bool Simulation_Running_LLG(State *state);
extern "C" bool Simulation_Running_GNEB(State *state);
extern "C" bool Simulation_Running_MMF(State *state);

#endif