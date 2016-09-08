#pragma once
#ifndef INTERFACE_PARAMETERS_H
#define INTERFACE_PARAMETERS_H
struct State;

// Set LLG
extern "C" void Parameters_Set_LLG_Time_Step(State *state, float dt, int idx_image=-1, int idx_chain=-1);
extern "C" void Parameters_Set_LLG_Damping(State *state, float damping, int idx_image=-1, int idx_chain=-1);
extern "C" void Parameters_Set_LLG_N_Iterations(State *state, int n_iterations, int idx_image=-1, int idx_chain=-1);
extern "C" void Parameters_Set_LLG_N_Iterations_Log(State *state, int n_iterations, int idx_image=-1, int idx_chain=-1);
// Set GNEB
extern "C" void Parameters_Set_GNEB_Spring_Constant(State *state, float spring_constant, int idx_image=-1, int idx_chain=-1);
extern "C" void Parameters_Set_GNEB_Climbing_Falling(State *state, bool climbing, bool falling, int idx_image=-1, int idx_chain=-1);
extern "C" void Parameters_Set_GNEB_N_Iterations(State *state, int n_iterations, int idx_image=-1, int idx_chain=-1);
extern "C" void Parameters_Set_GNEB_N_Iterations_Log(State *state, int n_iterations_log, int idx_image=-1, int idx_chain=-1);


// Get LLG
extern "C" void Parameters_Get_LLG_Time_Step(State *state, float * dt, int idx_image=-1, int idx_chain=-1);
extern "C" void Parameters_Get_LLG_Damping(State *state, float * damping, int idx_image=-1, int idx_chain=-1);
extern "C" int Parameters_Get_LLG_N_Iterations(State *state, int idx_image=-1, int idx_chain=-1);
extern "C" int Parameters_Get_LLG_N_Iterations_Log(State *state, int idx_image=-1, int idx_chain=-1);
// Set GNEB
extern "C" void Parameters_Get_GNEB_Spring_Constant(State *state, float * spring_constant, int idx_image=-1, int idx_chain=-1);
extern "C" void Parameters_Get_GNEB_Climbing_Falling(State *state, bool * climbing, bool * falling, int idx_image=-1, int idx_chain=-1);
extern "C" int Parameters_Get_GNEB_N_Iterations(State *state, int idx_image=-1, int idx_chain=-1);
extern "C" int Parameters_Get_GNEB_N_Iterations_Log(State *state, int idx_image=-1, int idx_chain=-1);

#endif