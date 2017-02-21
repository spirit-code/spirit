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

/*
	State_To_Config
	  Write a config file which should give the same state again when
	  used in State_Setup (modulo the number of chains and images)
*/
DLLEXPORT void State_To_Config(State * state, const char * config_file, const char * original_config_file="");

/*
	State_DateTime
	  Get the datetime tag (timepoint of creation) of this state
*/
DLLEXPORT const char * State_DateTime(State * state);

#include "DLL_Undefine_Export.h"
#endif