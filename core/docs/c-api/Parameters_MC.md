

MC Parameters
====================================================================

```C
#include "Spirit/Parameters_MC.h"
```



Set
--------------------------------------------------------------------



### Parameters_MC_Set_Output_Tag

```C
void Parameters_MC_Set_Output_Tag(State *state, const char * tag, int idx_image=-1, int idx_chain=-1)
```

Set the tag placed in front of output file names.

If the tag is "<time>", it will be the date-time of the creation of the state.



### Parameters_MC_Set_Output_Folder

```C
void Parameters_MC_Set_Output_Folder(State *state, const char * folder, int idx_image=-1, int idx_chain=-1)
```

Set the folder, where output files are placed.



### Parameters_MC_Set_Output_General

```C
void Parameters_MC_Set_Output_General(State *state, bool any, bool initial, bool final, int idx_image=-1, int idx_chain=-1)
```

Set whether to write any output files at all.



### Parameters_MC_Set_Output_Energy

```C
void Parameters_MC_Set_Output_Energy(State *state, bool energy_step, bool energy_archive, bool energy_spin_resolved, bool energy_divide_by_nos, bool energy_add_readability_lines, int idx_image=-1, int idx_chain=-1)
```

Set whether to write energy output files.

- `step`: whether to write a new file after each set of iterations
- `archive`: whether to append to an archive file after each set of iterations
- `spin_resolved`: whether to write a file containing the energy of each spin
- `divide_by_nos`: whether to divide energies by the number of spins
- `add_readability_lines`: whether to separate columns by lines



### Parameters_MC_Set_Output_Configuration

```C
void Parameters_MC_Set_Output_Configuration(State *state, bool configuration_step, bool configuration_archive, int configuration_filetype=IO_Fileformat_OVF_text, int idx_image=-1, int idx_chain=-1)
```

Set whether to write spin configuration output files.

- `step`: whether to write a new file after each set of iterations
- `archive`: whether to append to an archive file after each set of iterations
- `filetype`: the format in which the data is written



### Parameters_MC_Set_N_Iterations

```C
void Parameters_MC_Set_N_Iterations(State *state, int n_iterations, int n_iterations_log, int idx_image=-1, int idx_chain=-1)
```

Set the number of iterations and how often to log and write output.

- `n_iterations`: the maximum number of iterations
- `n_iterations_log`: the number of iterations after which status is logged and output written



Set Parameters
--------------------------------------------------------------------



### Parameters_MC_Set_Temperature

```C
void Parameters_MC_Set_Temperature(State *state, float T, int idx_image=-1, int idx_chain=-1)
```

Set the (homogeneous) base temperature [K].



### Parameters_MC_Set_Metropolis_Cone

```C
void Parameters_MC_Set_Metropolis_Cone(State *state, bool cone, float cone_angle, bool adaptive_cone, float target_acceptance_ratio, int idx_image=-1, int idx_chain=-1)
```

Configure the Metropolis parameters.

- use_cone: whether to displace the spins within a cone (otherwise: on the entire unit sphere)
- cone_angle: the opening angle within which the spin is placed
- use_adaptive_cone: automatically adapt the cone angle to achieve the set acceptance ratio
- target_acceptance_ratio: target acceptance ratio for the adaptive cone algorithm



### Parameters_MC_Set_Random_Sample

```C
void Parameters_MC_Set_Random_Sample(State *state, bool random_sample, int idx_image=-1, int idx_chain=-1)
```

Set whether spins should be sampled randomly or in sequence.



Get Output
--------------------------------------------------------------------



### Parameters_MC_Get_Output_Tag

```C
const char * Parameters_MC_Get_Output_Tag(State *state, int idx_image=-1, int idx_chain=-1)
```

Returns the output file tag.



### Parameters_MC_Get_Output_Folder

```C
const char * Parameters_MC_Get_Output_Folder(State *state, int idx_image=-1, int idx_chain=-1)
```

Returns the output folder.



### Parameters_MC_Get_Output_General

```C
void Parameters_MC_Get_Output_General(State *state, bool * any, bool * initial, bool * final, int idx_image=-1, int idx_chain=-1)
```

Retrieves whether to write any output at all.



### Parameters_MC_Get_Output_Energy

```C
void Parameters_MC_Get_Output_Energy(State *state, bool * energy_step, bool * energy_archive, bool * energy_spin_resolved, bool * energy_divide_by_nos, bool * energy_add_readability_lines, int idx_image=-1, int idx_chain=-1)
```

Retrieves the energy output settings.



### Parameters_MC_Get_Output_Configuration

```C
void Parameters_MC_Get_Output_Configuration(State *state, bool * configuration_step, bool * configuration_archive, int * configuration_filetype, int idx_image=-1, int idx_chain=-1)
```

Retrieves the spin configuration output settings.



### Parameters_MC_Get_N_Iterations

```C
void Parameters_MC_Get_N_Iterations(State *state, int * iterations, int * iterations_log, int idx_image=-1, int idx_chain=-1)
```

Returns the maximum number of iterations and the step size.



Get Parameters
--------------------------------------------------------------------



### Parameters_MC_Get_Temperature

```C
float Parameters_MC_Get_Temperature(State *state, int idx_image=-1, int idx_chain=-1)
```

Returns the global base temperature [K].



### Parameters_MC_Get_Metropolis_Cone

```C
void Parameters_MC_Get_Metropolis_Cone(State *state, bool * cone, float * cone_angle, bool * adaptive_cone, float * target_acceptance_ratio, int idx_image=-1, int idx_chain=-1)
```

Returns the Metropolis algorithm configuration.

- whether the spins are displaced within a cone (otherwise: on the entire unit sphere)
- the opening angle within which the spin is placed
- whether the cone angle is automatically adapted to achieve the set acceptance ratio
- target acceptance ratio for the adaptive cone algorithm



### Parameters_MC_Get_Random_Sample

```C
bool Parameters_MC_Get_Random_Sample(State *state, int idx_image=-1, int idx_chain=-1)
```

Returns whether spins should be sampled randomly or in sequence.

