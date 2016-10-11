#pragma once
#ifndef INTERFACE_STATE_H
#define INTERFACE_STATE_H
#include "DLL_Define_Export.h"

struct State;

/*
	State_Setup
	  Create the State and fill it with initial data
*/
DLLEXPORT State * State_Setup(const char * config_file = "");

/*
	State_Delete
	  Correctly deletes a State
*/
DLLEXPORT void State_Delete(State * state);

/*
	State_Update
      Update the state to hold current values
*/
void State_Update(State * state);

#include "DLL_Undefine_Export.h"
#endif