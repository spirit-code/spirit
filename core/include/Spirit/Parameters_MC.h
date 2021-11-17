#pragma once
#ifndef SPIRIT_CORE_PARAMETERS_MC_H
#define SPIRIT_CORE_PARAMETERS_MC_H
#include "IO.h"

#include "DLL_Define_Export.h"

struct State;

/*
MC Parameters
====================================================================

```C
#include "Spirit/Parameters_MC.h"
```
*/

/*
Set
--------------------------------------------------------------------
*/

/*
Set the tag placed in front of output file names.

If the tag is "<time>", it will be the date-time of the creation of the state.
*/
PREFIX void
Parameters_MC_Set_Output_Tag( State * state, const char * tag, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Set the folder, where output files are placed.
PREFIX void
Parameters_MC_Set_Output_Folder( State * state, const char * folder, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Set whether to write any output files at all.
PREFIX void Parameters_MC_Set_Output_General(
    State * state, bool any, bool initial, bool final, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Set whether to write energy output files.

- `step`: whether to write a new file after each set of iterations
- `archive`: whether to append to an archive file after each set of iterations
- `spin_resolved`: whether to write a file containing the energy of each spin
- `divide_by_nos`: whether to divide energies by the number of spins
- `add_readability_lines`: whether to separate columns by lines
*/
PREFIX void Parameters_MC_Set_Output_Energy(
    State * state, bool energy_step, bool energy_archive, bool energy_spin_resolved, bool energy_divide_by_nos,
    bool energy_add_readability_lines, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Set whether to write spin configuration output files.

- `step`: whether to write a new file after each set of iterations
- `archive`: whether to append to an archive file after each set of iterations
- `filetype`: the format in which the data is written
*/
PREFIX void Parameters_MC_Set_Output_Configuration(
    State * state, bool configuration_step, bool configuration_archive,
    int configuration_filetype = IO_Fileformat_OVF_text, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Set the number of iterations and how often to log and write output.

- `n_iterations`: the maximum number of iterations
- `n_iterations_log`: the number of iterations after which status is logged and output written
*/
PREFIX void Parameters_MC_Set_N_Iterations(
    State * state, int n_iterations, int n_iterations_log, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Set Parameters
--------------------------------------------------------------------
*/

// Set the (homogeneous) base temperature [K].
PREFIX void Parameters_MC_Set_Temperature( State * state, float T, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Configure the Metropolis parameters.

- use_cone: whether to displace the spins within a cone (otherwise: on the entire unit sphere)
- cone_angle: the opening angle within which the spin is placed
- use_adaptive_cone: automatically adapt the cone angle to achieve the set acceptance ratio
- target_acceptance_ratio: target acceptance ratio for the adaptive cone algorithm
*/
PREFIX void Parameters_MC_Set_Metropolis_Cone(
    State * state, bool cone, float cone_angle, bool adaptive_cone, float target_acceptance_ratio, int idx_image = -1,
    int idx_chain = -1 ) SUFFIX;

// Set whether spins should be sampled randomly or in sequence.
PREFIX void
Parameters_MC_Set_Random_Sample( State * state, bool random_sample, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Get Output
--------------------------------------------------------------------
*/

// Returns the output file tag.
PREFIX const char * Parameters_MC_Get_Output_Tag( State * state, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Returns the output folder.
PREFIX const char * Parameters_MC_Get_Output_Folder( State * state, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Retrieves whether to write any output at all.
PREFIX void Parameters_MC_Get_Output_General(
    State * state, bool * any, bool * initial, bool * final, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Retrieves the energy output settings.
PREFIX void Parameters_MC_Get_Output_Energy(
    State * state, bool * energy_step, bool * energy_archive, bool * energy_spin_resolved, bool * energy_divide_by_nos,
    bool * energy_add_readability_lines, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Retrieves the spin configuration output settings.
PREFIX void Parameters_MC_Get_Output_Configuration(
    State * state, bool * configuration_step, bool * configuration_archive, int * configuration_filetype,
    int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Returns the maximum number of iterations and the step size.
PREFIX void Parameters_MC_Get_N_Iterations(
    State * state, int * iterations, int * iterations_log, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Get Parameters
--------------------------------------------------------------------
*/

// Returns the global base temperature [K].
PREFIX float Parameters_MC_Get_Temperature( State * state, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Returns the Metropolis algorithm configuration.

- whether the spins are displaced within a cone (otherwise: on the entire unit sphere)
- the opening angle within which the spin is placed
- whether the cone angle is automatically adapted to achieve the set acceptance ratio
- target acceptance ratio for the adaptive cone algorithm
*/
PREFIX void Parameters_MC_Get_Metropolis_Cone(
    State * state, bool * cone, float * cone_angle, bool * adaptive_cone, float * target_acceptance_ratio,
    int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Returns whether spins should be sampled randomly or in sequence.
PREFIX bool Parameters_MC_Get_Random_Sample( State * state, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

#include "DLL_Undefine_Export.h"
#endif