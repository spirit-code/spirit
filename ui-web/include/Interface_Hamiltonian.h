#pragma once
#ifndef INTERFACE_HAMILTONIAN_H
#define INTERFACE_HAMILTONIAN_H

// Sets the Hamiltonian_Isotropic's parameters

extern "C" void Hamiltonian_Boundary_Conditions(bool a, bool b, bool c);

extern "C" void Hamiltonian_mu_s(float mu_s);

extern "C" void Hamiltonian_Field(float magnitude, float normal_x, float normal_y, float normal_z);

extern "C" void Hamiltonian_Exchange(float mu_s);

extern "C" void Hamiltonian_DMI(float mu_s);

extern "C" void Hamiltonian_Anisotropy(float mu_s);

extern "C" void Hamiltonian_STT(float mu_s);

extern "C" void Hamiltonian_Temperature(float mu_s);

#endif