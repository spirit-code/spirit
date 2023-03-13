#pragma once
#ifndef SPIRIT_CORE_QUANTITIES_H
#define SPIRIT_CORE_QUANTITIES_H
#include "Spirit_Defines.h"

#include "DLL_Define_Export.h"

struct State;

/*
Quantities
====================================================================

```C
#include "Spirit/Quantities.h"
```
*/

// Average spin direction
PREFIX void Quantity_Get_Average_Spin( State * state, scalar s[3], int idx_image = -1, int idx_chain = -1 );

// Total Magnetization
PREFIX void Quantity_Get_Magnetization( State * state, scalar m[3], int idx_image = -1, int idx_chain = -1 );

// Topological Charge
PREFIX scalar Quantity_Get_Topological_Charge( State * state, int idx_image = -1, int idx_chain = -1 );

// Topological Charge Density
PREFIX int Quantity_Get_Topological_Charge_Density(
    State * state, scalar * charge_density, int * triangle_indices, int idx_image = -1, int idx_chain = -1 );

// Minimum mode following information
PREFIX void Quantity_Get_Grad_Force_MinimumMode(
    State * state, scalar * gradient, scalar * eval, scalar * mode, scalar * forces, int idx_image = -1,
    int idx_chain = -1 );

#endif