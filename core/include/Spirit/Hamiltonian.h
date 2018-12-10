#pragma once
#ifndef INTERFACE_HAMILTONIAN_H
#define INTERFACE_HAMILTONIAN_H
#include "DLL_Define_Export.h"
struct State;

#define SPIRIT_CHIRALITY_BLOCH 1
#define SPIRIT_CHIRALITY_NEEL  2
#define SPIRIT_CHIRALITY_BLOCH_INVERSE -1
#define SPIRIT_CHIRALITY_NEEL_INVERSE  -2

#define SPIRIT_DDI_METHOD_NONE   0
#define SPIRIT_DDI_METHOD_FFT    1
#define SPIRIT_DDI_METHOD_FMM    2
#define SPIRIT_DDI_METHOD_CUTOFF 3

// Set the Hamiltonian's parameters
DLLEXPORT void Hamiltonian_Set_Boundary_Conditions(State *state, const bool* periodical, int idx_image=-1, int idx_chain=-1) noexcept;

DLLEXPORT void Hamiltonian_Set_Field(State *state, float magnitude, const float* normal, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Hamiltonian_Set_Anisotropy(State *state, float magnitude, const float* normal, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Hamiltonian_Set_Exchange(State *state, int n_shells, const float* jij, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Hamiltonian_Set_DMI(State *state, int n_shells, const float * dij, int chirality=SPIRIT_CHIRALITY_BLOCH, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Hamiltonian_Set_DDI(State *state, int ddi_method, int n_periodic_images[3], float cutoff_radius=0, int idx_image=-1, int idx_chain=-1) noexcept;

// Get the Hamiltonian's parameters
DLLEXPORT const char * Hamiltonian_Get_Name(State * state, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Hamiltonian_Get_Boundary_Conditions(State *state, bool * periodical, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Hamiltonian_Get_Field(State *state, float * magnitude, float * normal, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Hamiltonian_Get_Anisotropy(State *state, float * magnitude, float * normal, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Hamiltonian_Get_Exchange_Shells(State *state, int * n_shells, float * jij, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT int  Hamiltonian_Get_Exchange_N_Pairs(State *state, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Hamiltonian_Get_Exchange_Pairs(State *state, float * idx[2], float * translations[3], float * Jij, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Hamiltonian_Get_DMI_Shells(State *state, int * n_shells, float * dij, int * chirality, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT int  Hamiltonian_Get_DMI_N_Pairs(State *state, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Hamiltonian_Get_DDI(State *state, int * ddi_method, int n_periodic_images[3], float * cutoff_radius, int idx_image=-1, int idx_chain=-1) noexcept;

#include "DLL_Undefine_Export.h"
#endif