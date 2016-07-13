#pragma once
#ifndef INTERFACE_HAMILTONIAN_H
#define INTERFACE_HAMILTONIAN_H
struct State;
// Sets the Hamiltonian_Isotropic's parameters

extern "C" void Hamiltonian_Boundary_Conditions(State *state, bool a, bool b, bool c);

extern "C" void Hamiltonian_mu_s(State *state, float mu_s);

extern "C" void Hamiltonian_Field(State *state, float magnitude, float normal_x, float normal_y, float normal_z);

extern "C" void Hamiltonian_Exchange(State *state, int n_shells, float* jij);

extern "C" void Hamiltonian_DMI(State *state, float dij);

extern "C" void Hamiltonian_Anisotropy(State *state, float magnitude, float normal_x, float normal_y, float normal_z);

extern "C" void Hamiltonian_STT(State *state, float magnitude, float normal_x, float normal_y, float normal_z);

extern "C" void Hamiltonian_Temperature(State *state, float t);

#endif