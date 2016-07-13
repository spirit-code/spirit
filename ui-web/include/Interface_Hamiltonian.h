#pragma once
#ifndef INTERFACE_HAMILTONIAN_H
#define INTERFACE_HAMILTONIAN_H
struct State;

// Set the Hamiltonian's parameters
extern "C" void Hamiltonian_Set_Boundary_Conditions(State *state, bool periodical_a, bool periodical_b, bool periodical_c);
extern "C" void Hamiltonian_Set_mu_s(State *state, float mu_s);
extern "C" void Hamiltonian_Set_Field(State *state, float magnitude, float normal_x, float normal_y, float normal_z);
extern "C" void Hamiltonian_Set_Exchange(State *state, int n_shells, float* jij);
extern "C" void Hamiltonian_Set_DMI(State *state, float dij);
extern "C" void Hamiltonian_Set_Anisotropy(State *state, float magnitude, float normal_x, float normal_y, float normal_z);
extern "C" void Hamiltonian_Set_STT(State *state, float magnitude, float normal_x, float normal_y, float normal_z);
extern "C" void Hamiltonian_Set_Temperature(State *state, float T);

// Get the Hamiltonian's parameters
extern "C" void Hamiltonian_Get_Boundary_Conditions(State *state, bool * periodical_a, bool * periodical_b, bool * periodical_c);
extern "C" void Hamiltonian_Get_mu_s(State *state, float * mu_s);
extern "C" void Hamiltonian_Get_Field(State *state, float * magnitude, float * normal_x, float * normal_y, float * normal_z);
extern "C" void Hamiltonian_Get_Exchange(State *state, int * n_shells, float * jij);
extern "C" void Hamiltonian_Get_DMI(State *state, float * dij);
extern "C" void Hamiltonian_Get_Anisotropy(State *state, float * magnitude, float * normal_x, float * normal_y, float * normal_z);
extern "C" void Hamiltonian_Get_STT(State *state, float * magnitude, float * normal_x, float * normal_y, float * normal_z);
extern "C" void Hamiltonian_Get_Temperature(State *state, float * T);


#endif