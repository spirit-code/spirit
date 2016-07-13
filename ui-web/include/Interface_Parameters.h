#pragma once
#ifndef INTERFACE_PARAMETERS_H
#define INTERFACE_PARAMETERS_H
struct State;

// Set LLG
extern "C" void Parameters_Set_LLG_Time_Step(State *state, float dt);
extern "C" void Parameters_Set_LLG_Damping(State *state, float damping);
// Set GNEB
extern "C" void Parameters_Set_GNEB_Spring_Constant(State *state, float spring_constant);
extern "C" void Parameters_Set_GNEB_Climbing_Falling(State *state, bool climbing, bool falling);


// Get LLG
extern "C" void Parameters_Get_LLG_Time_Step(State *state, float * dt);
extern "C" void Parameters_Get_LLG_Damping(State *state, float * damping);
// Set GNEB
extern "C" void Parameters_Get_GNEB_Spring_Constant(State *state, float * spring_constant);
extern "C" void Parameters_Get_GNEB_Climbing_Falling(State *state, bool * climbing, bool * falling);

#endif