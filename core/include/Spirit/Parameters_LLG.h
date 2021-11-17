#pragma once
#ifndef SPIRIT_CORE_PARAMETERS_LLG_H
#define SPIRIT_CORE_PARAMETERS_LLG_H
#include "IO.h"

#include "DLL_Define_Export.h"

struct State;

/*
LLG Parameters
====================================================================

```C
#include "Spirit/Parameters_LLG.h"
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
Parameters_LLG_Set_Output_Tag( State * state, const char * tag, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Set the folder, where output files are placed.
PREFIX void
Parameters_LLG_Set_Output_Folder( State * state, const char * folder, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Set whether to write any output files at all.
PREFIX void Parameters_LLG_Set_Output_General(
    State * state, bool any, bool initial, bool final, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Set whether to write energy output files.

- `step`: whether to write a new file after each set of iterations
- `archive`: whether to append to an archive file after each set of iterations
- `spin_resolved`: whether to write a file containing the energy of each spin
- `divide_by_nos`: whether to divide energies by the number of spins
- `add_readability_lines`: whether to separate columns by lines
*/
PREFIX void Parameters_LLG_Set_Output_Energy(
    State * state, bool energy_step, bool energy_archive, bool energy_spin_resolved, bool energy_divide_by_nos,
    bool energy_add_readability_lines, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Set whether to write spin configuration output files.

- `step`: whether to write a new file after each set of iterations
- `archive`: whether to append to an archive file after each set of iterations
- `filetype`: the format in which the data is written
*/
PREFIX void Parameters_LLG_Set_Output_Configuration(
    State * state, bool configuration_step, bool configuration_archive,
    int configuration_filetype = IO_Fileformat_OVF_text, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Set the number of iterations and how often to log and write output.

- `n_iterations`: the maximum number of iterations
- `n_iterations_log`: the number of iterations after which status is logged and output written
*/
PREFIX void Parameters_LLG_Set_N_Iterations(
    State * state, int n_iterations, int n_iterations_log, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Set Parameters
--------------------------------------------------------------------
*/

/*
Set whether to minimise the energy without precession.

This only influences dynamics solvers, which will then perform pseudodynamics,
simulating only the damping part of the LLG equation.
*/
PREFIX void
Parameters_LLG_Set_Direct_Minimization( State * state, bool direct, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Set the convergence limit.

When the maximum absolute component value of the force drops below this value,
the calculation is considered converged and will stop.
*/
PREFIX void
Parameters_LLG_Set_Convergence( State * state, float convergence, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Set the time step [ps] for the calculation.
PREFIX void Parameters_LLG_Set_Time_Step( State * state, float dt, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Set the Gilbert damping parameter [unitless].
PREFIX void Parameters_LLG_Set_Damping( State * state, float damping, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Set the spin current configuration.

- use_gradient: `True`: use the spatial gradient, `False`: monolayer approximation
- magnitude: current strength
- direction: current direction or polarisation direction, array of shape (3)
*/
PREFIX void Parameters_LLG_Set_STT(
    State * state, bool use_gradient, float magnitude, const float normal[3], int idx_image = -1,
    int idx_chain = -1 ) SUFFIX;

// Set the (homogeneous) base temperature [K].
PREFIX void Parameters_LLG_Set_Temperature( State * state, float T, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Set an additional temperature gradient.

- gradient_inclination: inclination of the temperature gradient [K/a]
- gradient_direction: direction of the temperature gradient, array of shape (3)
*/
PREFIX void Parameters_LLG_Set_Temperature_Gradient(
    State * state, float inclination, const float direction[3], int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Get Output
--------------------------------------------------------------------
*/

// Returns the output file tag.
PREFIX const char * Parameters_LLG_Get_Output_Tag( State * state, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Returns the output folder.
PREFIX const char * Parameters_LLG_Get_Output_Folder( State * state, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Retrieves whether to write any output at all.
PREFIX void Parameters_LLG_Get_Output_General(
    State * state, bool * any, bool * initial, bool * final, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Retrieves the energy output settings.
PREFIX void Parameters_LLG_Get_Output_Energy(
    State * state, bool * energy_step, bool * energy_archive, bool * energy_spin_resolved, bool * energy_divide_by_nos,
    bool * energy_add_readability_lines, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Retrieves the spin configuration output settings.
PREFIX void Parameters_LLG_Get_Output_Configuration(
    State * state, bool * configuration_step, bool * configuration_archive, int * configuration_filetype,
    int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Returns the maximum number of iterations and the step size.
PREFIX void Parameters_LLG_Get_N_Iterations(
    State * state, int * iterations, int * iterations_log, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Get Parameters
--------------------------------------------------------------------
*/

// Returns whether only energy minimisation will be performed.
PREFIX bool Parameters_LLG_Get_Direct_Minimization( State * state, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Returns the convergence value.
PREFIX float Parameters_LLG_Get_Convergence( State * state, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Returns the time step [ps].
PREFIX float Parameters_LLG_Get_Time_Step( State * state, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Returns the Gilbert damping parameter.
PREFIX float Parameters_LLG_Get_Damping( State * state, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Returns the global base temperature [K].
PREFIX float Parameters_LLG_Get_Temperature( State * state, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Retrieves the temperature gradient.

- inclination of the temperature gradient [K/a]
- direction of the temperature gradient, array of shape (3)
*/
PREFIX void Parameters_LLG_Get_Temperature_Gradient(
    State * state, float * inclination, float direction[3], int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Returns the spin current configuration.

- magnitude
- direction, array of shape (3)
- whether the spatial gradient is used
*/
PREFIX void Parameters_LLG_Get_STT(
    State * state, bool * use_gradient, float * magnitude, float normal[3], int idx_image = -1,
    int idx_chain = -1 ) SUFFIX;

#include "DLL_Undefine_Export.h"
#endif