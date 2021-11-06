

System
====================================================================

```C
#include "Spirit/System.h"
```

Spin systems are often referred to as "images" throughout Spirit.
The `idx_image` is used throughout the API to specify which system
out of the chain a function should be applied to.
`idx_image=-1` refers to the active image of the chain.



### System_Get_Index

```C
int System_Get_Index(State * state)
```

Returns the index of the currently active spin system in the chain.



### System_Get_NOS

```C
int System_Get_NOS(State * state, int idx_image=-1, int idx_chain=-1)
```

Returns the number of spins (NOS) of a spin system.



### System_Get_Spin_Directions

```C
scalar * System_Get_Spin_Directions(State * state, int idx_image=-1, int idx_chain=-1)
```

Returns a pointer to the spin orientations data.

The array is contiguous and of shape (NOS, 3).



### System_Get_Effective_Field

```C
scalar * System_Get_Effective_Field(State * state, int idx_image=-1, int idx_chain=-1)
```

Returns a pointer to the effective field data.

The array is contiguous and of shape (NOS, 3).



### System_Get_Eigenmode

```C
scalar * System_Get_Eigenmode(State * state, int idx_mode, int idx_image=-1, int idx_chain=-1)
```

Returns a pointer to the data of the N'th eigenmode of a spin system.

The array is contiguous and of shape (NOS, 3).



### System_Get_Rx

```C
float System_Get_Rx(State * state, int idx_image=-1, int idx_chain=-1)
```

Returns the reaction coordinate of a system along the chain.



### System_Get_Energy

```C
float System_Get_Energy(State * state, int idx_image=-1, int idx_chain=-1)
```

Returns the energy of a spin system.



### System_Get_Energy_Array_Names

```C
int System_Get_Energy_Array_Names(State * state, char* names, int idx_image=-1, int idx_chain=-1)
```

If 'names' is a nullptr, the required length of the char array is returned.



### System_Get_Energy_Array

```C
int System_Get_Energy_Array(State * state, float * energies, bool divide_by_nspins=true, int idx_image=-1, int idx_chain=-1)
```

If 'energies' is a nullptr, the required length of the energies array is returned.



### System_Get_Eigenvalues

```C
void System_Get_Eigenvalues(State * state, float * eigenvalues, int idx_image=-1, int idx_chain=-1)
```

Retrieves the eigenvalues of a spin system



### System_Print_Energy_Array

```C
void System_Print_Energy_Array(State * state, int idx_image=-1, int idx_chain=-1)
```

Write the energy as formatted output to the console



### System_Update_Data

```C
void System_Update_Data(State * state, int idx_image=-1, int idx_chain=-1)
```

Update Data (primarily for plots)



### System_Update_Eigenmodes

```C
void System_Update_Eigenmodes(State *state, int idx_image=-1, int idx_chain=-1)
```

Update Eigenmodes (primarily for visualisation or saving)

