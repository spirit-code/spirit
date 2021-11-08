#pragma once
#ifndef SPIRIT_CORE_PARAMETERS_GNEB_H
#define SPIRIT_CORE_PARAMETERS_GNEB_H
#include "IO.h"

#include "DLL_Define_Export.h"

struct State;

/*
GNEB Parameters
====================================================================

```C
#include "Spirit/Parameters_GNEB.h"
```
*/

// Regular GNEB image type.
#define GNEB_IMAGE_NORMAL 0

/*
Climbing GNEB image type.
Climbing images move towards maxima along the path.
*/
#define GNEB_IMAGE_CLIMBING 1

/*
Falling GNEB image type.
Falling images move towards the closest minima.
*/
#define GNEB_IMAGE_FALLING 2

/*
Stationary GNEB image type.
Stationary images are not influenced during a GNEB calculation.
*/
#define GNEB_IMAGE_STATIONARY 3

/*
Set Output
--------------------------------------------------------------------
*/

/*
Set the tag placed in front of output file names.

If the tag is "<time>", it will be the date-time of the creation of the state.
*/
PREFIX void Parameters_GNEB_Set_Output_Tag( State * state, const char * tag, int idx_chain = -1 ) SUFFIX;

// Set the folder, where output files are placed.
PREFIX void Parameters_GNEB_Set_Output_Folder( State * state, const char * folder, int idx_chain = -1 ) SUFFIX;

// Set whether to write any output files at all.
PREFIX void
Parameters_GNEB_Set_Output_General( State * state, bool any, bool initial, bool final, int idx_chain = -1 ) SUFFIX;

/*
Set whether to write energy output files.

- `step`: whether to write a new file after each set of iterations
- `interpolated`: whether to write a file containing interpolated reaction coordinate and energy values
- `divide_by_nos`: whether to divide energies by the number of spins
- `add_readability_lines`: whether to separate columns by lines
*/
PREFIX void Parameters_GNEB_Set_Output_Energies(
    State * state, bool step, bool interpolated, bool divide_by_nos, bool add_readability_lines,
    int idx_chain = -1 ) SUFFIX;

/*
Set whether to write chain output files.

- `step`: whether to write a new file after each set of iterations
- `filetype`: the format in which the data is written
*/
PREFIX void Parameters_GNEB_Set_Output_Chain(
    State * state, bool step, int filetype = IO_Fileformat_OVF_text, int idx_chain = -1 ) SUFFIX;

/*
Set the number of iterations and how often to log and write output.

- `n_iterations`: the maximum number of iterations
- `n_iterations_log`: the number of iterations after which status is logged and output written
*/
PREFIX void
Parameters_GNEB_Set_N_Iterations( State * state, int n_iterations, int n_iterations_log, int idx_chain = -1 ) SUFFIX;

/*
Set Parameters
--------------------------------------------------------------------
*/

/*
Set the convergence limit.

When the maximum absolute component value of the force drops below this value,
the calculation is considered converged and will stop.
*/
PREFIX void
Parameters_GNEB_Set_Convergence( State * state, float convergence, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Set the spring force constant.
PREFIX void Parameters_GNEB_Set_Spring_Constant(
    State * state, float spring_constant, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Set the ratio between energy and reaction coordinate.
PREFIX void Parameters_GNEB_Set_Spring_Force_Ratio( State * state, float ratio, int idx_chain = -1 ) SUFFIX;

// Set the path shortening constant.
PREFIX void Parameters_GNEB_Set_Path_Shortening_Constant(
    State * state, float path_shortening_constant, int idx_chain = -1 ) SUFFIX;

// Set the GNEB image type (see the integers defined above).
PREFIX void
Parameters_GNEB_Set_Climbing_Falling( State * state, int image_type, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Automatically set GNEB image types.

Minima along the path will be set to falling, maxima to climbing and the rest to regular.
*/
PREFIX void Parameters_GNEB_Set_Image_Type_Automatically( State * state, int idx_chain = -1 ) SUFFIX;

// Returns the maximum number of iterations and the step size.
PREFIX void Parameters_GNEB_Set_N_Energy_Interpolations( State * state, int n, int idx_chain = -1 ) SUFFIX;

/*
Get Output
--------------------------------------------------------------------
*/

// Returns the output file tag.
PREFIX const char * Parameters_GNEB_Get_Output_Tag( State * state, int idx_chain = -1 ) SUFFIX;

// Returns the output folder.
PREFIX const char * Parameters_GNEB_Get_Output_Folder( State * state, int idx_chain = -1 ) SUFFIX;

// Retrieves whether to write any output at all.
PREFIX void Parameters_GNEB_Get_Output_General(
    State * state, bool * any, bool * initial, bool * final, int idx_chain = -1 ) SUFFIX;

// Retrieves the energy output settings.
PREFIX void Parameters_GNEB_Get_Output_Energies(
    State * state, bool * step, bool * interpolated, bool * divide_by_nos, bool * add_readability_lines,
    int idx_chain = -1 ) SUFFIX;

// Retrieves the chain output settings.
PREFIX void Parameters_GNEB_Get_Output_Chain( State * state, bool * step, int * filetype, int idx_chain = -1 ) SUFFIX;

// Returns the maximum number of iterations and the step size.
PREFIX void
Parameters_GNEB_Get_N_Iterations( State * state, int * iterations, int * iterations_log, int idx_chain = -1 ) SUFFIX;

/*
Get Parameters
--------------------------------------------------------------------
*/

// Simulation Parameters
PREFIX float Parameters_GNEB_Get_Convergence( State * state, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Returns the spring force constant.
PREFIX float Parameters_GNEB_Get_Spring_Constant( State * state, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Returns the spring force cratio of energy to reaction coordinate.
PREFIX float Parameters_GNEB_Get_Spring_Force_Ratio( State * state, int idx_chain = -1 ) SUFFIX;

// Return the path shortening constant.
PREFIX float Parameters_GNEB_Get_Path_Shortening_Constant( State * state, int idx_chain = -1 ) SUFFIX;

/*
Returns the integer of whether an image is regular, climbing, falling, or stationary.

The integers are defined above.
*/
PREFIX int Parameters_GNEB_Get_Climbing_Falling( State * state, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Returns the number of energy values interpolated between images.
PREFIX int Parameters_GNEB_Get_N_Energy_Interpolations( State * state, int idx_chain = -1 ) SUFFIX;

#include "DLL_Undefine_Export.h"
#endif