#pragma once
#ifndef SPIRIT_CORE_TRANSITIONS_H
#define SPIRIT_CORE_TRANSITIONS_H
#include "DLL_Define_Export.h"

struct State;

/*
Transitions
====================================================================

```C
#include "Spirit/Transitions.h"
```

Setting transitions between spin configurations over a chain.
*/

/*
A linear interpolation between two spin configurations on a chain.

The spins are moved along great circles connecting the start and end
points, making it the shortest possible connection path between the
two configurations.

- `idx_1`: the index of the first image
- `idx_2`: the index of the second image. `idx_2 > idx_1` is required
*/
PREFIX void Transition_Homogeneous( State * state, int idx_1, int idx_2, int idx_chain = -1 ) SUFFIX;

/*
A helper function that makes the chain denser by inserting interpolated images between all images.

- `n_interpolate`: the number of images to be inserted between to adjaced images, n_interpolate=1 nearly doubles the
length of the chain
*/
PREFIX void Transition_Homogeneous_Insert_Interpolated( State * state, int n_interpolate, int idx_chain = -1 ) SUFFIX;

/*
Adds some stochastic noise to the transition between two images.

- `temperature`: a measure of the intensity of the noise
- `idx_1`: the index of the first image
- `idx_2`: the index of the second image. `idx_2 > idx_1` is required
*/
PREFIX void
Transition_Add_Noise_Temperature( State * state, float temperature, int idx_1, int idx_2, int idx_chain = -1 ) SUFFIX;

#include "DLL_Undefine_Export.h"
#endif