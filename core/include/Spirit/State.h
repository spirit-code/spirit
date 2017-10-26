#pragma once
#ifndef INTERFACE_STATE_H
#define INTERFACE_STATE_H
#include "DLL_Define_Export.h"

struct State;

/*
	State_Setup
	  Create the State and fill it with initial data
*/
DLLEXPORT State * State_Setup(const char * config_file = "", bool quiet = false) noexcept;

/*
	State_Delete
	  Correctly deletes a State
*/
DLLEXPORT void State_Delete(State * state) noexcept;

/*
	State_Update
      Update the state to hold current values
*/
DLLEXPORT void State_Update(State * state) noexcept;

/*
	State_To_Config
	  Write a config file which should give the same state again when
	  used in State_Setup (modulo the number of chains and images)
*/
DLLEXPORT void State_To_Config(State * state, const char * config_file, const char * original_config_file="") noexcept;

/*
	State_DateTime
	  Get the datetime tag (timepoint of creation) of this state
*/
DLLEXPORT const char * State_DateTime(State * state) noexcept;

#include "DLL_Undefine_Export.h"
#endif