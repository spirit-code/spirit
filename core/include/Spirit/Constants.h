#pragma once
#ifndef INTERFACE_CONSTANTS_H
#define INTERFACE_CONSTANTS_H
#include "DLL_Define_Export.h"

#include "Spirit_Defines.h"

// The Bohr Magneton [meV / T]
DLLEXPORT scalar Constants_mu_B();
// Gyromagnetic ratio of electron [rad/(s*T)]
DLLEXPORT scalar Constants_gamma();
// The Boltzmann constant [meV / K]
DLLEXPORT scalar Constants_k_B();

#include "DLL_Undefine_Export.h"
#endif