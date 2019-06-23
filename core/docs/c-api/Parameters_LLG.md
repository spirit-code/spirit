

LLG Parameters
====================================================================

```C
#include "Spirit/Parameters_LLG.h"
```



Set Output
--------------------------------------------------------------------



### Parameters_LLG_Set_Output_Tag

```C
void Parameters_LLG_Set_Output_Tag(State *state, const char * tag, int idx_image=-1, int idx_chain=-1)
```

Set the tag placed in front of output file names.

If the tag is "<time>", it will be the date-time of the creation of the state.



### Parameters_LLG_Set_Output_Folder

```C
void Parameters_LLG_Set_Output_Folder(State *state, const char * folder, int idx_image=-1, int idx_chain=-1)
```

Set the folder, where output files are placed.



### Parameters_LLG_Set_Output_General

```C
void Parameters_LLG_Set_Output_General(State *state, bool any, bool initial, bool final, int idx_image=-1, int idx_chain=-1)
```

Set whether to write any output files at all.



### Parameters_LLG_Set_Output_Energy

```C
void Parameters_LLG_Set_Output_Energy(State *state, bool energy_step, bool energy_archive, bool energy_spin_resolved, bool energy_divide_by_nos, bool energy_add_readability_lines, int idx_image=-1, int idx_chain=-1)
```

Set whether to write energy output files.

- `step`: whether to write a new file after each set of iterations
- `archive`: whether to append to an archive file after each set of iterations
- `spin_resolved`: whether to write a file containing the energy of each spin
- `divide_by_nos`: whether to divide energies by the number of spins
- `add_readability_lines`: whether to separate columns by lines



### Parameters_LLG_Set_Output_Configuration

```C
void Parameters_LLG_Set_Output_Configuration(State *state, bool configuration_step, bool configuration_archive, int configuration_filetype=IO_Fileformat_OVF_text, int idx_image=-1, int idx_chain=-1)
```

Set whether to write spin configuration output files.

- `step`: whether to write a new file after each set of iterations
- `archive`: whether to append to an archive file after each set of iterations
- `filetype`: the format in which the data is written



### Parameters_LLG_Set_N_Iterations

```C
void Parameters_LLG_Set_N_Iterations(State *state, int n_iterations, int n_iterations_log, int idx_image=-1, int idx_chain=-1)
```

Set the number of iterations and how often to log and write output.

- `n_iterations`: the maximum number of iterations
- `n_iterations_log`: the number of iterations after which status is logged and output written



Set Parameters
--------------------------------------------------------------------



### Parameters_LLG_Set_Direct_Minimization

```C
void Parameters_LLG_Set_Direct_Minimization(State *state, bool direct, int idx_image=-1, int idx_chain=-1)
```

Set whether to minimise the energy without precession.

This only influences dynamics solvers, which will then perform pseudodynamics,
simulating only the damping part of the LLG equation.



### Parameters_LLG_Set_Convergence

```C
void Parameters_LLG_Set_Convergence(State *state, float convergence, int idx_image=-1, int idx_chain=-1)
```

Set the convergence limit.

When the maximum absolute component value of the force drops below this value,
the calculation is considered converged and will stop.



### Parameters_LLG_Set_Time_Step

```C
void Parameters_LLG_Set_Time_Step(State *state, float dt, int idx_image=-1, int idx_chain=-1)
```

Set the time step [ps] for the calculation.



### Parameters_LLG_Set_Damping

```C
void Parameters_LLG_Set_Damping(State *state, float damping, int idx_image=-1, int idx_chain=-1)
```

Set the Gilbert damping parameter [unitless].



### Parameters_LLG_Set_STT

```C
void Parameters_LLG_Set_STT(State *state, bool use_gradient, float magnitude, const float normal[3], int idx_image=-1, int idx_chain=-1)
```

Set the spin current configuration.

- use_gradient: `True`: use the spatial gradient, `False`: monolayer approximation
- magnitude: current strength
- direction: current direction or polarisation direction, array of shape (3)



### Parameters_LLG_Set_Temperature

```C
void Parameters_LLG_Set_Temperature(State *state, float T, int idx_image=-1, int idx_chain=-1)
```

Set the (homogeneous) base temperature [K].



### Parameters_LLG_Set_Temperature_Gradient

```C
void Parameters_LLG_Set_Temperature_Gradient(State *state, float inclination, const float direction[3], int idx_image=-1, int idx_chain=-1)
```

Set an additional temperature gradient.

- gradient_inclination: inclination of the temperature gradient [K/a]
- gradient_direction: direction of the temperature gradient, array of shape (3)



Get Output
--------------------------------------------------------------------



### Parameters_LLG_Get_Output_Tag

```C
const char * Parameters_LLG_Get_Output_Tag(State *state, int idx_image=-1, int idx_chain=-1)
```

Returns the output file tag.



### Parameters_LLG_Get_Output_Folder

```C
const char * Parameters_LLG_Get_Output_Folder(State *state, int idx_image=-1, int idx_chain=-1)
```

Returns the output folder.



### Parameters_LLG_Get_Output_General

```C
void Parameters_LLG_Get_Output_General(State *state, bool * any, bool * initial, bool * final, int idx_image=-1, int idx_chain=-1)
```

Retrieves whether to write any output at all.



### Parameters_LLG_Get_Output_Energy

```C
void Parameters_LLG_Get_Output_Energy(State *state, bool * energy_step, bool * energy_archive, bool * energy_spin_resolved, bool * energy_divide_by_nos, bool * energy_add_readability_lines, int idx_image=-1, int idx_chain=-1)
```

Retrieves the energy output settings.



### Parameters_LLG_Get_Output_Configuration

```C
void Parameters_LLG_Get_Output_Configuration(State *state, bool * configuration_step, bool * configuration_archive, int * configuration_filetype, int idx_image=-1, int idx_chain=-1)
```

Retrieves the spin configuration output settings.



### Parameters_LLG_Get_N_Iterations

```C
void Parameters_LLG_Get_N_Iterations(State *state, int * iterations, int * iterations_log, int idx_image=-1, int idx_chain=-1)
```

Returns the maximum number of iterations and the step size.



Get Parameters
--------------------------------------------------------------------



### Parameters_LLG_Get_Direct_Minimization

```C
bool Parameters_LLG_Get_Direct_Minimization(State *state, int idx_image=-1, int idx_chain=-1)
```

Returns whether only energy minimisation will be performed.



### Parameters_LLG_Get_Convergence

```C
float Parameters_LLG_Get_Convergence(State *state, int idx_image=-1, int idx_chain=-1)
```

Returns the convergence value.



### Parameters_LLG_Get_Time_Step

```C
float Parameters_LLG_Get_Time_Step(State *state, int idx_image=-1, int idx_chain=-1)
```

Returns the time step [ps].



### Parameters_LLG_Get_Damping

```C
float Parameters_LLG_Get_Damping(State *state, int idx_image=-1, int idx_chain=-1)
```

Returns the Gilbert damping parameter.



### Parameters_LLG_Get_Temperature

```C
float Parameters_LLG_Get_Temperature(State *state, int idx_image=-1, int idx_chain=-1)
```

Returns the global base temperature [K].



### Parameters_LLG_Get_Temperature_Gradient

```C
void Parameters_LLG_Get_Temperature_Gradient(State *state, float * direction, float normal[3], int idx_image=-1, int idx_chain=-1)
```

Retrieves the temperature gradient.

- inclination of the temperature gradient [K/a]
- direction of the temperature gradient, array of shape (3)



### Parameters_LLG_Get_STT

```C
void Parameters_LLG_Get_STT(State *state, bool * use_gradient, float * magnitude, float normal[3], int idx_image=-1, int idx_chain=-1)
```

Returns the spin current configuration.

- magnitude
- direction, array of shape (3)
- whether the spatial gradient is used

