#pragma once
#ifndef INTERFACE_CONSTANTS_H
#define INTERFACE_CONSTANTS_H
#include "DLL_Define_Export.h"

#include "Spirit_Defines.h"

/*
Constants
====================================================================

```C
#include "Spirit/Constants.h"
```

Physical constants in units compatible to what is used in Spirit.
*/

// The Bohr Magneton [meV / T]
PREFIX scalar Constants_mu_B() SUFFIX;

// The vacuum permeability [T^2 m^3 / meV]
PREFIX scalar Constants_mu_0() SUFFIX;

// The Boltzmann constant [meV / K]
PREFIX scalar Constants_k_B() SUFFIX;

// Planck constant [meV*ps / rad]
PREFIX scalar Constants_hbar() SUFFIX;

// Millirydberg [mRy / meV]
PREFIX scalar Constants_mRy() SUFFIX;

// Gyromagnetic ratio of electron [rad / (s*T)]
PREFIX scalar Constants_gamma() SUFFIX;

// Electron g-factor [unitless]
PREFIX scalar Constants_g_e() SUFFIX;

// Erg [erg / meV]
PREFIX scalar Constants_erg() SUFFIX;

// Joule [J / meV]
PREFIX scalar Constants_Joule() SUFFIX;

// Pi [rad]
PREFIX scalar Constants_Pi() SUFFIX;

// Pi/2 [rad]
PREFIX scalar Constants_Pi_2() SUFFIX;

#include "DLL_Undefine_Export.h"
#endif