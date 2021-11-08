#pragma once
#ifndef SPIRIT_CORE_QUANTITIES_H
#define SPIRIT_CORE_QUANTITIES_H
#include "DLL_Define_Export.h"

struct State;

/*
Quantities
====================================================================

```C
#include "Spirit/Quantities.h"
```
*/

// Total Magnetization
PREFIX void Quantity_Get_Magnetization( State * state, float m[3], int idx_image = -1, int idx_chain = -1 );

// Topological Charge
PREFIX float Quantity_Get_Topological_Charge( State * state, int idx_image = -1, int idx_chain = -1 );

// Minimum mode following information
PREFIX void Quantity_Get_Grad_Force_MinimumMode(
    State * state, float * gradient, float * eval, float * mode, float * forces, int idx_image = -1,
    int idx_chain = -1 );

#endif