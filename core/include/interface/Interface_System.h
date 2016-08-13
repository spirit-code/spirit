#pragma once
#ifndef INTERFACE_SYSTEM_H
#define INTERFACE_SYSTEM_H
struct State;

// Info
extern "C" int System_Get_Index(State * state);
extern "C" int System_Get_NOS(State * state, int idx_image=-1, int idx_chain=-1);

extern "C" double * System_Get_Spin_Directions(State * state, int idx_image=-1, int idx_chain=-1);

#endif