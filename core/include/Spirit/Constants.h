#pragma once
#ifndef INTERFACE_CONSTANTS_H
#define INTERFACE_CONSTANTS_H
#include "DLL_Define_Export.h"

#include "Spirit_Defines.h"

// The Bohr Magneton [meV / T]
DLLEXPORT scalar Constants_mu_B() noexcept;
// The Boltzmann constant [meV / K]
DLLEXPORT scalar Constants_k_B() noexcept;
// Planck constant [meV*ps / rad]
DLLEXPORT scalar Constants_hbar() noexcept;
// Millirydberg [mRy / meV]
DLLEXPORT scalar Constants_mRy() noexcept;
// Gyromagnetic ratio of electron [rad / (s*T)]
DLLEXPORT scalar Constants_gamma() noexcept;
// Electron g-factor [unitless]
DLLEXPORT scalar Constants_g_e() noexcept;

#include "DLL_Undefine_Export.h"
#endif