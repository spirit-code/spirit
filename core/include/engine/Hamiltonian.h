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


#include <vector>

#include "Logging.h"
#include "Exception.h"

namespace Engine
{
	/*
		The Hamiltonian contains the interaction parameters of a System.
		It also defines the functions to calculate the Effective Field and Energy.
	*/
	class Hamiltonian
	{
	public:
		Hamiltonian(std::vector<bool> boundary_conditions) : boundary_conditions(boundary_conditions) {};

		// Calculate the Effective field of a spin configuration
		virtual void Effective_Field(const std::vector<double> & spins, std::vector<double> & field)
		{
			// Not Implemented!
			Log(Utility::Log_Level::Error, Utility::Log_Sender::All, std::string("Tried to use Hamiltonian::Effective_Field() of the Hamiltonian base class!"));
			throw Utility::Exception::Not_Implemented;
		};

		// Calculate the Energy of a spin configuration
		virtual double Energy(std::vector<double> & spins)
		{
			// Not Implemented!
			Log(Utility::Log_Level::Error, Utility::Log_Sender::All, std::string("Tried to use Hamiltonian::Energy() of the Hamiltonian base class!"));
			throw Utility::Exception::Not_Implemented;
			return 0.0;
		}

		// Calculate the Energies of the spins of a configuration
		virtual std::vector<std::vector<double>> Energy_Array_per_Spin(std::vector<double> & spins)
		{
			Log(Utility::Log_Level::Error, Utility::Log_Sender::All, std::string("Tried to use Hamiltonian::Energy_Array_per_Spin() of the Hamiltonian base class!"));
			throw Utility::Exception::Not_Implemented;
			return std::vector<std::vector<double>>(spins.size(), std::vector<double>(7, 0.0));
		}

		// Calculate the Effective Field of a spin configuration
		virtual std::vector<double> Energy_Array(std::vector<double> & spins)
		{
			Log(Utility::Log_Level::Error, Utility::Log_Sender::All, std::string("Tried to use Hamiltonian::Energy_Array() of the Hamiltonian base class!"));
			throw Utility::Exception::Not_Implemented;
			return std::vector<double>(7, 0.0);
		}

		// Boundary conditions
		std::vector<bool> boundary_conditions; // [3] (a, b, c)

	};
}
#endif