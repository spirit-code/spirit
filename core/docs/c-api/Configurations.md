

Configurations
====================================================================

```C
#include "Spirit/Configurations.h"
```

Setting spin configurations for individual spin systems.

The position of the relative center and a set of conditions can be defined.



Clipboard
--------------------------------------------------------------------



### Configuration_To_Clipboard

```C
void Configuration_To_Clipboard(State *state, int idx_image=-1, int idx_chain=-1)
```

Copies the current spin configuration to the clipboard



### Configuration_From_Clipboard

```C
void Configuration_From_Clipboard(State *state, const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false, int idx_image=-1, int idx_chain=-1)
```

Pastes the clipboard spin configuration



### Configuration_From_Clipboard_Shift

```C
bool Configuration_From_Clipboard_Shift(State *state, const float shift[3], const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted = false, int idx_image=-1, int idx_chain=-1)
```

Pastes the clipboard spin configuration



Nonlocalised
--------------------------------------------------------------------



### Configuration_Domain

```C
void Configuration_Domain(State *state, const float direction[3], const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false, int idx_image=-1, int idx_chain=-1)
```

Creates a homogeneous domain



### Configuration_PlusZ

```C
void Configuration_PlusZ(State *state, const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false, int idx_image=-1, int idx_chain=-1)
```

Points all spins in +z direction



### Configuration_MinusZ

```C
void Configuration_MinusZ(State *state, const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false, int idx_image=-1, int idx_chain=-1)
```

Points all spins in -z direction



### Configuration_Random

```C
void Configuration_Random(State *state, const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false, bool external=false, int idx_image=-1, int idx_chain=-1)
```

Points all spins in random directions



### Configuration_SpinSpiral

```C
void Configuration_SpinSpiral(State *state, const char * direction_type, float q[3], float axis[3], float theta, const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false, int idx_image=-1, int idx_chain=-1)
```

Spin spiral



### Configuration_SpinSpiral_2q

```C
void Configuration_SpinSpiral_2q(State *state, const char * direction_type, float q1[3], float q2[3], float axis[3], float theta, const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false, int idx_image=-1, int idx_chain=-1)
```

2q spin spiral



Perturbations
--------------------------------------------------------------------



### Configuration_Add_Noise_Temperature

```C
void Configuration_Add_Noise_Temperature(State *state, float temperature, const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false, int idx_image=-1, int idx_chain=-1)
```

Adds some random noise scaled by temperature



### Configuration_Displace_Eigenmode

```C
void Configuration_Displace_Eigenmode(State *state, int idx_mode, int idx_image=-1, int idx_chain=-1)
```

Calculate the eigenmodes of the system (Image)



Localised
--------------------------------------------------------------------



### Configuration_Skyrmion

```C
void Configuration_Skyrmion(State *state, float r, float order, float phase, bool upDown, bool achiral, bool rl, const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false, int idx_image=-1, int idx_chain=-1)
```

Create a skyrmion configuration



### Configuration_DW_Skyrmion

```C
void Configuration_DW_Skyrmion(State *state, float dw_radius, float dw_width, float order, float phase, bool upDown, bool achiral, bool rl, const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false, int idx_image=-1, int idx_chain=-1)
```

Create a skyrmion configuration with the circular domain wall ("swiss knife") profile



### Configuration_Hopfion

```C
void Configuration_Hopfion(State *state, float r, int order=1, const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false,  int idx_image=-1, int idx_chain=-1)
```

Create a toroidal Hopfion



Pinning and atom types
--------------------------------------------------------------------

This API can also be used to change the `pinned` state and the `atom type`
of atoms in a spacial region, instead of using translation indices.



### Configuration_Set_Pinned

```C
void Configuration_Set_Pinned(State *state, bool pinned, const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false, int idx_image=-1, int idx_chain=-1)
```

Pinning



### Configuration_Set_Atom_Type

```C
void Configuration_Set_Atom_Type(State *state, int type, const float position[3]=defaultPos, const float r_cut_rectangular[3]=defaultRect, float r_cut_cylindrical=-1, float r_cut_spherical=-1, bool inverted=false, int idx_image=-1, int idx_chain=-1)
```

Atom types

