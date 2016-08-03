#pragma once
#ifndef INTERFACE_TRANSITIONS_H
#define INTERFACE_TRANSITIONS_H
struct State;

extern "C" void Transition_Homogeneous(State *state, int idx_1, int idx_2, int idx_chain=-1);

#endif