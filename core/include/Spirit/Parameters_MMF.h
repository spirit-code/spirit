#pragma once
#ifndef SPIRIT_CORE_PARAMETERS_MMF_H
#define SPIRIT_CORE_PARAMETERS_MMF_H
#include "IO.h"

#include "DLL_Define_Export.h"

struct State;

/*
MMF Parameters
====================================================================

```C
#include "Spirit/Parameters_MMF.h"
```
*/

/*
Set Output
--------------------------------------------------------------------
*/

/*
Set the tag placed in front of output file names.

If the tag is "<time>", it will be the date-time of the creation of the state.
*/
PREFIX void
Parameters_MMF_Set_Output_Tag( State * state, const char * tag, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Set the folder, where output files are placed.
PREFIX void
Parameters_MMF_Set_Output_Folder( State * state, const char * folder, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Set whether to write any output files at all.
PREFIX void Parameters_MMF_Set_Output_General(
    State * state, bool any, bool initial, bool final, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Set whether to write energy output files.

- `step`: whether to write a new file after each set of iterations
- `archive`: whether to append to an archive file after each set of iterations
- `spin_resolved`: whether to write a file containing the energy of each spin
- `divide_by_nos`: whether to divide energies by the number of spins
- `add_readability_lines`: whether to separate columns by lines
*/
PREFIX void Parameters_MMF_Set_Output_Energy(
    State * state, bool step, bool archive, bool spin_resolved, bool divide_by_nos, bool add_readability_lines,
    int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Set whether to write spin configuration output files.

- `step`: whether to write a new file after each set of iterations
- `archive`: whether to append to an archive file after each set of iterations
- `filetype`: the format in which the data is written
*/
PREFIX void Parameters_MMF_Set_Output_Configuration(
    State * state, bool step, bool archive, int filetype, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Set the number of iterations and how often to log and write output.

- `n_iterations`: the maximum number of iterations
- `n_iterations_log`: the number of iterations after which status is logged and output written
*/
PREFIX void Parameters_MMF_Set_N_Iterations(
    State * state, int n_iterations, int n_iterations_log, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Set Parameters
--------------------------------------------------------------------
*/

// Set the number of modes to be calculated at each iteration.
PREFIX void Parameters_MMF_Set_N_Modes( State * state, int n_modes, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Set the index of the mode to follow.
PREFIX void
Parameters_MMF_Set_N_Mode_Follow( State * state, int n_mode_follow, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Get Output
--------------------------------------------------------------------
*/

// Returns the output file tag.
PREFIX const char * Parameters_MMF_Get_Output_Tag( State * state, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Returns the output folder.
PREFIX const char * Parameters_MMF_Get_Output_Folder( State * state, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Retrieves whether to write any output at all.
PREFIX void Parameters_MMF_Get_Output_General(
    State * state, bool * any, bool * initial, bool * final, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Retrieves the energy output settings.
PREFIX void Parameters_MMF_Get_Output_Energy(
    State * state, bool * step, bool * archive, bool * spin_resolved, bool * divide_by_nos,
    bool * add_readability_lines, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Retrieves the spin configuration output settings.
PREFIX void Parameters_MMF_Get_Output_Configuration(
    State * state, bool * step, bool * archive, int * filetype, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Returns the maximum number of iterations and the step size.
PREFIX void Parameters_MMF_Get_N_Iterations(
    State * state, int * iterations, int * iterations_log, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Get Parameters
--------------------------------------------------------------------
*/

// Returns the number of modes calculated at each iteration.
PREFIX int Parameters_MMF_Get_N_Modes( State * state, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Returns the index of the mode which to follow.
PREFIX int Parameters_MMF_Get_N_Mode_Follow( State * state, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

#include "DLL_Undefine_Export.h"
#endif