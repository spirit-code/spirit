#pragma once
#ifndef INTERFACE_TRANSITIONS_H
#define INTERFACE_TRANSITIONS_H
struct State;

extern "C" void Transition_Homogeneous(State *state, int idx_1, int idx_2, int idx_chain=-1);
extern "C" void Transition_Add_Noise_Temperature(State *state, double temperature, int idx_1, int idx_2, int idx_chain=-1);

#endif