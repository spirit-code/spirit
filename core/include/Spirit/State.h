#pragma once
#ifndef INTERFACE_STATE_H
#define INTERFACE_STATE_H
#include "DLL_Define_Export.h"

struct State;

/*
    State_Setup
      Create the State and fill it with initial data
*/
PREFIX State * State_Setup(const char * config_file = "", bool quiet = false) SUFFIX;

/*
    State_Delete
      Correctly deletes a State
*/
PREFIX void State_Delete(State * state) SUFFIX;

/*
    State_Update
      Update the state to hold current values
*/
PREFIX void State_Update(State * state) SUFFIX;

/*
    State_To_Config
      Write a config file which should give the same state again when
      used in State_Setup (modulo the number of chains and images)
*/
PREFIX void State_To_Config(State * state, const char * config_file, const char * original_config_file="") SUFFIX;

/*
    State_DateTime
      Get the datetime tag (timepoint of creation) of this state
*/
PREFIX const char * State_DateTime(State * state) SUFFIX;

#include "DLL_Undefine_Export.h"
#endif