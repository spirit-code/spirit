#pragma once
#ifndef INTERFACE_COLLECTION_H
#define INTERFACE_COLLECTION_H
struct State;

// Info
extern "C" int Collection_Get_NOC(State * state);

// Move
extern "C" void Collection_next_Chain(State * state);
extern "C" void Collection_prev_Chain(State * state);

#endif