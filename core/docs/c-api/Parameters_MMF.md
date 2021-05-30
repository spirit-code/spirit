

MMF Parameters
====================================================================

```C
#include "Spirit/Parameters_MMF.h"
```



Set Output
--------------------------------------------------------------------



### Parameters_MMF_Set_Output_Tag

```C
void Parameters_MMF_Set_Output_Tag(State *state, const char * tag, int idx_image=-1, int idx_chain=-1)
```

Set the tag placed in front of output file names.

If the tag is "<time>", it will be the date-time of the creation of the state.



### Parameters_MMF_Set_Output_Folder

```C
void Parameters_MMF_Set_Output_Folder(State *state, const char * folder, int idx_image=-1, int idx_chain=-1)
```

Set the folder, where output files are placed.



### Parameters_MMF_Set_Output_General

```C
void Parameters_MMF_Set_Output_General(State *state, bool any, bool initial, bool final, int idx_image=-1, int idx_chain=-1)
```

Set whether to write any output files at all.



### Parameters_MMF_Set_Output_Energy

```C
void Parameters_MMF_Set_Output_Energy(State *state, bool step, bool archive, bool spin_resolved, bool divide_by_nos, bool add_readability_lines, int idx_image=-1, int idx_chain=-1)
```

Set whether to write energy output files.

- `step`: whether to write a new file after each set of iterations
- `archive`: whether to append to an archive file after each set of iterations
- `spin_resolved`: whether to write a file containing the energy of each spin
- `divide_by_nos`: whether to divide energies by the number of spins
- `add_readability_lines`: whether to separate columns by lines



### Parameters_MMF_Set_Output_Configuration

```C
void Parameters_MMF_Set_Output_Configuration(State *state, bool step, bool archive, int filetype, int idx_image=-1, int idx_chain=-1)
```

Set whether to write spin configuration output files.

- `step`: whether to write a new file after each set of iterations
- `archive`: whether to append to an archive file after each set of iterations
- `filetype`: the format in which the data is written



### Parameters_MMF_Set_N_Iterations

```C
void Parameters_MMF_Set_N_Iterations(State *state, int n_iterations, int n_iterations_log, int idx_image=-1, int idx_chain=-1)
```

Set the number of iterations and how often to log and write output.

- `n_iterations`: the maximum number of iterations
- `n_iterations_log`: the number of iterations after which status is logged and output written



Set Parameters
--------------------------------------------------------------------



### Parameters_MMF_Set_N_Modes

```C
void Parameters_MMF_Set_N_Modes(State *state, int n_modes, int idx_image=-1, int idx_chain=-1)
```

Set the number of modes to be calculated at each iteration.



### Parameters_MMF_Set_N_Mode_Follow

```C
void Parameters_MMF_Set_N_Mode_Follow(State *state, int n_mode_follow, int idx_image=-1, int idx_chain=-1)
```

Set the index of the mode to follow.



Get Output
--------------------------------------------------------------------



### Parameters_MMF_Get_Output_Tag

```C
const char * Parameters_MMF_Get_Output_Tag(State *state, int idx_image=-1, int idx_chain=-1)
```

Returns the output file tag.



### Parameters_MMF_Get_Output_Folder

```C
const char * Parameters_MMF_Get_Output_Folder(State *state, int idx_image=-1, int idx_chain=-1)
```

Returns the output folder.



### Parameters_MMF_Get_Output_General

```C
void Parameters_MMF_Get_Output_General(State *state, bool * any, bool * initial, bool * final, int idx_image=-1, int idx_chain=-1)
```

Retrieves whether to write any output at all.



### Parameters_MMF_Get_Output_Energy

```C
void Parameters_MMF_Get_Output_Energy(State *state, bool * step, bool * archive, bool * spin_resolved, bool * divide_by_nos, bool * add_readability_lines, int idx_image=-1, int idx_chain=-1)
```

Retrieves the energy output settings.



### Parameters_MMF_Get_Output_Configuration

```C
void Parameters_MMF_Get_Output_Configuration(State *state, bool * step, bool * archive, int * filetype, int idx_image=-1, int idx_chain=-1)
```

Retrieves the spin configuration output settings.



### Parameters_MMF_Get_N_Iterations

```C
void Parameters_MMF_Get_N_Iterations(State *state, int * iterations, int * iterations_log, int idx_image=-1, int idx_chain=-1)
```

Returns the maximum number of iterations and the step size.



Get Parameters
--------------------------------------------------------------------



### Parameters_MMF_Get_N_Modes

```C
int Parameters_MMF_Get_N_Modes(State *state, int idx_image=-1, int idx_chain=-1)
```

Returns the number of modes calculated at each iteration.



### Parameters_MMF_Get_N_Mode_Follow

```C
int Parameters_MMF_Get_N_Mode_Follow(State *state, int idx_image=-1, int idx_chain=-1)
```

Returns the index of the mode which to follow.

