#pragma once
#ifndef INTERFACE_Transitions_H
#define INTERFACE_Transitions_H
struct State;

extern "C" void Transition_Homogeneous(State *state, int idx_chain=-1);

#endif