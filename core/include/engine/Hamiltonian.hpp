#pragma once
#ifndef HAMILTONIAN_H
#define HAMILTONIAN_H


// Defines for the Positions of Energy Contributions inside the Energy Array.
// This way we may arbitrarily rearrange the array or add more interactions without rewriting code.
#define ENERGY_POS_ZEEMAN 0
#define ENERGY_POS_ANISOTROPY 1
#define ENERGY_POS_EXCHANGE 2
#define ENERGY_POS_DMI 3
#define ENERGY_POS_BQC 4
#define ENERGY_POS_FSC 5
#define ENERGY_POS_DD 6


#include <random>
#include <vector>

#include "Core_Defines.h"
#include "Logging.hpp"
#include "Exception.hpp"

namespace Engine
{
	/*
		The Hamiltonian contains the interaction parameters of a System.
		It also defines the functions to calculate the Effective Field and Energy.
	*/
	class Hamiltonian
	{
	public:
		Hamiltonian(std::vector<bool> boundary_conditions);

		/*
			Calculate the Hessian matrix of a spin configuration.
			This function uses finite differences and may thus be quite inefficient. You should
			override it if you want to get proper performance.
			This function is the fallback for derived classes where it has not been overridden.
		*/
		virtual void Hessian(const std::vector<scalar> & spins, std::vector<scalar> & hessian);

		/*
			Calculate the effective field of a spin configuration.
			This function uses finite differences and may thus be quite inefficient. You should
			override it if you want to get proper performance.
			This function is the fallback for derived classes where it has not been overridden.
		*/
		virtual void Effective_Field(const std::vector<scalar> & spins, std::vector<scalar> & field);

		// Calculate the Energy of a spin configuration
		virtual scalar Energy(const std::vector<scalar> & spins);

		// Calculate the Energies of the spins of a configuration
		virtual std::vector<std::vector<scalar>> Energy_Array_per_Spin(const std::vector<scalar> & spins);

		// Calculate the Effective Field of a spin configuration
		virtual std::vector<scalar> Energy_Array(const std::vector<scalar> & spins);

		// Hamiltonian name as string
		virtual const std::string& Name();

		// Boundary conditions
		std::vector<bool> boundary_conditions; // [3] (a, b, c)
	
	private:
		std::mt19937 prng;
		std::uniform_int_distribution<int> distribution_int;
		scalar delta;
	};
}
#endif