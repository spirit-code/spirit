

EMA Parameters
====================================================================

```C
#include "Spirit/Parameters_EMA.h"
```

This method, if needed, calculates modes (they can also be read in from a file)
and perturbs the spin system periodically in the direction of the eigenmode.



Set
--------------------------------------------------------------------



### Parameters_EMA_Set_N_Modes

```C
void Parameters_EMA_Set_N_Modes(State *state, int n_modes, int idx_image=-1, int idx_chain=-1)
```

Set the number of modes to calculate or use.



### Parameters_EMA_Set_N_Mode_Follow

```C
void Parameters_EMA_Set_N_Mode_Follow(State *state, int n_mode_follow, int idx_image=-1, int idx_chain=-1)
```

Set the index of the mode to use.



### Parameters_EMA_Set_Frequency

```C
void Parameters_EMA_Set_Frequency(State *state, float frequency, int idx_image=-1, int idx_chain=-1)
```

Set the frequency with which the mode is applied.



### Parameters_EMA_Set_Amplitude

```C
void Parameters_EMA_Set_Amplitude(State *state, float amplitude, int idx_image=-1, int idx_chain=-1)
```

Set the amplitude with which the mode is applied.



### Parameters_EMA_Set_Snapshot

```C
void Parameters_EMA_Set_Snapshot(State *state, bool snapshot, int idx_image=-1, int idx_chain=-1)
```

Set whether to displace the system statically instead of periodically.



Get
--------------------------------------------------------------------



### Parameters_EMA_Get_N_Modes

```C
int Parameters_EMA_Get_N_Modes(State *state, int idx_image=-1, int idx_chain=-1)
```

Returns the number of modes to calculate or use.



### Parameters_EMA_Get_N_Mode_Follow

```C
int Parameters_EMA_Get_N_Mode_Follow(State *state, int idx_image=-1, int idx_chain=-1)
```

Returns the index of the mode to use.



### Parameters_EMA_Get_Frequency

```C
float Parameters_EMA_Get_Frequency(State *state, int idx_image=-1, int idx_chain=-1)
```

Returns the frequency with which the mode is applied.



### Parameters_EMA_Get_Amplitude

```C
float Parameters_EMA_Get_Amplitude(State *state, int idx_image=-1, int idx_chain=-1)
```

Returns the amplitude with which the mode is applied.



### Parameters_EMA_Get_Snapshot

```C
bool Parameters_EMA_Get_Snapshot(State *state, int idx_image=-1, int idx_chain=-1)
```

Returns whether to displace the system statically instead of periodically.

