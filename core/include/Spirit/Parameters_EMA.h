#pragma once
#ifndef SPIRIT_CORE_PARAMETERS_EMA_H
#define SPIRIT_CORE_PARAMETERS_EMA_H
#include "IO.h"

#include "DLL_Define_Export.h"

struct State;

/*
EMA Parameters
====================================================================

```C
#include "Spirit/Parameters_EMA.h"
```

This method, if needed, calculates modes (they can also be read in from a file)
and perturbs the spin system periodically in the direction of the eigenmode.
*/

/*
Set
--------------------------------------------------------------------
*/

// Set the number of modes to calculate or use.
PREFIX void Parameters_EMA_Set_N_Modes( State * state, int n_modes, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Set the index of the mode to use.
PREFIX void
Parameters_EMA_Set_N_Mode_Follow( State * state, int n_mode_follow, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Set the frequency with which the mode is applied.
PREFIX void
Parameters_EMA_Set_Frequency( State * state, float frequency, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Set the amplitude with which the mode is applied.
PREFIX void
Parameters_EMA_Set_Amplitude( State * state, float amplitude, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Set whether to displace the system statically instead of periodically.
PREFIX void Parameters_EMA_Set_Snapshot( State * state, bool snapshot, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Get
--------------------------------------------------------------------
*/

// Returns the number of modes to calculate or use.
PREFIX int Parameters_EMA_Get_N_Modes( State * state, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Returns the index of the mode to use.
PREFIX int Parameters_EMA_Get_N_Mode_Follow( State * state, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Returns the frequency with which the mode is applied.
PREFIX float Parameters_EMA_Get_Frequency( State * state, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Returns the amplitude with which the mode is applied.
PREFIX float Parameters_EMA_Get_Amplitude( State * state, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Returns whether to displace the system statically instead of periodically.
PREFIX bool Parameters_EMA_Get_Snapshot( State * state, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

#include "DLL_Undefine_Export.h"
#endif