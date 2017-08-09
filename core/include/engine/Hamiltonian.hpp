#pragma once
#ifndef HAMILTONIAN_H
#define HAMILTONIAN_H

#include <random>
#include <vector>

#include "Spirit_Defines.h"
#include <engine/Vectormath_Defines.hpp>

namespace Engine
{
	/*
		The Hamiltonian contains the interaction parameters of a System.
		It also defines the functions to calculate the Effective Field and Energy.
	*/
	class Hamiltonian
	{
	public:
		Hamiltonian(intfield boundary_conditions);

		/*
			Update the Energy array.
			This needs to be done every time the parameters are changed, in case an energy
			contribution is now non-zero or vice versa.
		*/
		virtual void Update_Energy_Contributions();

		/*
			Calculate the Hessian matrix of a spin configuration.
			This function uses finite differences and may thus be quite inefficient. You should
			override it if you want to get proper performance.
			This function is the fallback for derived classes where it has not been overridden.
		*/
		virtual void Hessian(const vectorfield & spins, MatrixX & hessian);

		/*
			Calculate the Hessian matrix of a spin configuration.
			This function uses finite differences and may thus be quite inefficient.
		*/
		virtual void Hessian_FD(const vectorfield & spins, MatrixX & hessian) final;
		
		/*
			Calculate the energy gradient of a spin configuration.
			This function uses finite differences and may thus be quite inefficient. You should
			override it if you want to get proper performance.
			This function is the fallback for derived classes where it has not been overridden.
		*/
		virtual void Gradient(const vectorfield & spins, vectorfield & gradient);

		/*
			Calculate the energy gradient of a spin configuration.
			This function uses finite differences and may thus be quite inefficient.
		*/
		virtual void Gradient_FD(const vectorfield & spins, vectorfield & gradient) final;

		// Calculate the Energy contributions for the spins of a configuration
		virtual void Energy_Contributions_per_Spin(const vectorfield & spins, std::vector<std::pair<std::string, scalarfield>> & contributions);

		// Calculate the Energy contributions for a spin configuration
		virtual std::vector<std::pair<std::string, scalar>> Energy_Contributions(const vectorfield & spins);

		// Calculate the Energy of a spin configuration
		virtual scalar Energy(const vectorfield & spins);

		// Hamiltonian name as string
		virtual const std::string& Name();

		// Boundary conditions
		intfield boundary_conditions; // [3] (a, b, c)
	
	protected:
		// Check atom types
		inline bool check_atom_type(int atom_type)
		{
			#ifdef SPIRIT_ENABLE_DEFECTS
				// If defects are enabled we check for
				//		vacancies (type < 0)
				if (atom_type >= 0) return true;
				else return false;
			#else
				// Else we just return true
				return true;
			#endif
		}
		inline bool check_atom_type(int atom_type, int reference_type)
		{
			#ifdef SPIRIT_ENABLE_DEFECTS
				// If defects are enabled we do a check if
				//		atom types match.
				if (atom_type == reference_type) return true;
				else return false;
			#else
				// Else we just return true
				return true;
			#endif
		}

		// Energy contributions per spin
		std::vector<std::pair<std::string, scalarfield>> energy_contributions_per_spin;

		std::mt19937 prng;
		std::uniform_int_distribution<int> distribution_int;
		scalar delta;
	};
}
#endif