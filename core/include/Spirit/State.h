#pragma once
#ifndef SPIRIT_CORE_STATE_H
#define SPIRIT_CORE_STATE_H
#include "DLL_Define_Export.h"

/*
State
====================================================================

```C
#include "Spirit/State.h"
```

To create a new state with one chain containing a single image,
initialized by an [input file](Input.md), and run the most simple example
of a **spin dynamics simulation**:

```C
#import "Spirit/Simulation.h"
#import "Spirit/State.h"

const char * cfgfile = "input/input.cfg";  // Input file
State * p_state = State_Setup(cfgfile);    // State setup
Simulation_LLG_Start(p_state, Solver_SIB); // Start a LLG simulation using the SIB solver
State_Delete(p_state)                      // State cleanup
```
*/

/*
The opaque state struct, containing all calculation data.

```
+-----------------------------+
| State                       |
| +-------------------------+ |
| | Chain                   | |
| | +--------------------+  | |
| | | 0th System ≡ Image |  | |
| | +--------------------+  | |
| | +--------------------+  | |
| | | 1st System ≡ Image |  | |
| | +--------------------+  | |
| |   .                     | |
| |   .                     | |
| |   .                     | |
| | +--------------------+  | |
| | | Nth System ≡ Image |  | |
| | +--------------------+  | |
| +-------------------------+ |
+-----------------------------+
```

This is passed to and is operated on by the API functions.

A new state can be created with `State_Setup()`, where you can pass
a [config file](Input.md) specifying your initial system parameters.
If you do not pass a config file, the implemented defaults are used.
**Note that you currently cannot change the geometry of the systems
in your state once they are initialized.**
*/
struct State;

/*
Create the State and fill it with initial data.

- `config_file`: if a config file is given, it will be parsed for
  keywords specifying the initial values. Otherwise, defaults are used
- `quiet`: if `true`, the defaults are changed such that only very few
  messages will be printed to the console and no output files are written
*/
PREFIX State * State_Setup( const char * config_file = "", bool quiet = false ) SUFFIX;

/*
Correctly deletes a State and frees the corresponding memory.
*/
PREFIX void State_Delete( State * state ) SUFFIX;

/*
Update the state to hold current values.
*/
PREFIX void State_Update( State * state ) SUFFIX;

/*
Write a config file which should give the same state again when
used in State_Setup (modulo the number of chains and images)
*/
PREFIX void State_To_Config( State * state, const char * config_file, const char * comment = "" ) SUFFIX;

/*
Returns a string containing the datetime tag (timepoint of creation) of this state.

Format: `yyyy-mm-dd_hh-mm-ss`
*/
PREFIX const char * State_DateTime( State * state ) SUFFIX;

#include "DLL_Undefine_Export.h"
#endif