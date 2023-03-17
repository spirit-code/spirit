

Hamiltonian
====================================================================

```C
#include "Spirit/Hamiltonian.h"
```

This currently only provides an interface to the Heisenberg Hamiltonian.



DMI chirality
--------------------------------------------------------------------

This means that
- Bloch chirality corresponds to DM vectors along bonds
- Neel chirality corresponds to DM vectors orthogonal to bonds

Neel chirality should therefore only be used in 2D systems.



### SPIRIT_CHIRALITY_BLOCH

```C
SPIRIT_CHIRALITY_BLOCH 1
```

Bloch chirality



### SPIRIT_CHIRALITY_NEEL

```C
SPIRIT_CHIRALITY_NEEL  2
```

Neel chirality



### SPIRIT_CHIRALITY_BLOCH_INVERSE

```C
SPIRIT_CHIRALITY_BLOCH_INVERSE -1
```

Bloch chirality, inverted DM vectors



### SPIRIT_CHIRALITY_NEEL_INVERSE

```C
SPIRIT_CHIRALITY_NEEL_INVERSE  -2
```

Neel chirality, inverted DM vectors



Dipole-Dipole method
--------------------------------------------------------------------



### SPIRIT_DDI_METHOD_NONE

```C
SPIRIT_DDI_METHOD_NONE   0
```

Do not use dipolar interactions



### SPIRIT_DDI_METHOD_FFT

```C
SPIRIT_DDI_METHOD_FFT    1
```

Use fast Fourier transform (FFT) convolutions



### SPIRIT_DDI_METHOD_FMM

```C
SPIRIT_DDI_METHOD_FMM    2
```

Use the fast multipole method (FMM)



### SPIRIT_DDI_METHOD_CUTOFF

```C
SPIRIT_DDI_METHOD_CUTOFF 3
```

Use a direct summation with a cutoff radius



Setters
--------------------------------------------------------------------



### Hamiltonian_Set_Boundary_Conditions

```C
void Hamiltonian_Set_Boundary_Conditions(State *state, const bool* periodical, int idx_image=-1, int idx_chain=-1)
```

Set the boundary conditions along the translation directions [a, b, c]



### Hamiltonian_Set_Field

```C
void Hamiltonian_Set_Field(State *state, float magnitude, const float* normal, int idx_image=-1, int idx_chain=-1)
```

Set the (homogeneous) external magnetic field [T]



### Hamiltonian_Set_Anisotropy

```C
void Hamiltonian_Set_Anisotropy(State *state, float magnitude, const float* normal, int idx_image=-1, int idx_chain=-1)
```

Set a global uniaxial anisotropy [meV]



### Hamiltonian_Set_Exchange

```C
void Hamiltonian_Set_Exchange(State *state, int n_shells, const float* jij, int idx_image=-1, int idx_chain=-1)
```

Set the exchange interaction in terms of neighbour shells [meV]



### Hamiltonian_Set_DMI

```C
void Hamiltonian_Set_DMI(State *state, int n_shells, const float * dij, int chirality=SPIRIT_CHIRALITY_BLOCH, int idx_image=-1, int idx_chain=-1)
```

Set the Dzyaloshinskii-Moriya interaction in terms of neighbour shells [meV]



### Hamiltonian_Set_DDI

```C
void Hamiltonian_Set_DDI(State *state, int ddi_method, int n_periodic_images[3], float cutoff_radius=0, bool pb_zero_padding=true, int idx_image=-1, int idx_chain=-1)
```

Configure the dipole-dipole interaction

- `ddi_method`: see integers defined above
- `n_periodic_images`: how many repetition of the spin configuration to
  append along the translation directions [a, b, c], if periodical
  boundary conditions are used
- `cutoff_radius`: the distance at which to stop the direct summation,
  if used
- `pb_zero_padding`: if `True` zero padding is used even for periodical directions



Getters
--------------------------------------------------------------------



### Hamiltonian_Get_Name

```C
const char * Hamiltonian_Get_Name(State * state, int idx_image=-1, int idx_chain=-1)
```

Returns a string containing the name of the Hamiltonian in use



### Hamiltonian_Get_Boundary_Conditions

```C
void Hamiltonian_Get_Boundary_Conditions(State *state, bool * periodical, int idx_image=-1, int idx_chain=-1)
```

Retrieves the boundary conditions



### Hamiltonian_Get_Field

```C
void Hamiltonian_Get_Field(State *state, float * magnitude, float * normal, int idx_image=-1, int idx_chain=-1)
```

Retrieves the external magnetic field [T]



### Hamiltonian_Get_Anisotropy

```C
void Hamiltonian_Get_Anisotropy(State *state, float * magnitude, float * normal, int idx_image=-1, int idx_chain=-1)
```

Retrieves the uniaxial anisotropy [meV]



### Hamiltonian_Get_Exchange_Shells

```C
void Hamiltonian_Get_Exchange_Shells(State *state, int * n_shells, float * jij, int idx_image=-1, int idx_chain=-1)
```

Retrieves the exchange interaction in terms of neighbour shells.

**Note:** if the interactions were specified as pairs, this function
will retrieve `n_shells=0`.



### Hamiltonian_Get_Exchange_N_Pairs

```C
int  Hamiltonian_Get_Exchange_N_Pairs(State *state, int idx_image=-1, int idx_chain=-1)
```

Returns the number of exchange interaction pairs



### Hamiltonian_Get_DMI_Shells

```C
void Hamiltonian_Get_DMI_Shells(State *state, int * n_shells, float * dij, int * chirality, int idx_image=-1, int idx_chain=-1)
```

Retrieves the Dzyaloshinskii-Moriya interaction in terms of neighbour shells.

**Note:** if the interactions were specified as pairs, this function
will retrieve `n_shells=0`.



### Hamiltonian_Get_DMI_N_Pairs

```C
int  Hamiltonian_Get_DMI_N_Pairs(State *state, int idx_image=-1, int idx_chain=-1)
```

Returns the number of Dzyaloshinskii-Moriya interaction pairs



### Hamiltonian_Get_DDI

```C
void Hamiltonian_Get_DDI(State *state, int * ddi_method, int n_periodic_images[3], float * cutoff_radius,  bool * pb_zero_padding, int idx_image=-1, int idx_chain=-1)
```

Retrieves the dipole-dipole interaction configuration.

- `ddi_method`: see integers defined above
- `n_periodic_images`: how many repetitions of the spin configuration to
  append along the translation directions [a, b, c], if periodical boundary conditions are used
- `cutoff_radius`: the distance at which to stop the direct summation, if method_cutoff is used
- `pb_zero_padding`: if `True` zero padding is used even for periodical directions

