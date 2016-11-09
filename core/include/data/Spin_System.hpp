#pragma once
#ifndef DATA_SPIN_SYSTEM_H
#define DATA_SPIN_SYSTEM_H

#include <random>
#include <thread>
#include <memory>

#include "Core_Defines.h"
#include "data/Geometry.hpp"
#include "engine/Hamiltonian.hpp"
#include "engine/Hamiltonian_Isotropic.hpp"
#include "engine/Hamiltonian_Anisotropic.hpp"
#include "engine/Hamiltonian_Gaussian.hpp"
#include "data/Parameters_Method_LLG.hpp"
#include "data/Parameters_Method_GNEB.hpp"

namespace Data
{
	/*
	Spin_System contains all setup information on one system (one set of spins, one image).
	This includes: Spin positions and orientations, Neighbours, Interaction constants, System parameters
	*/
	class Spin_System
	{
	public:
		// Constructor
		Spin_System(std::unique_ptr<Engine::Hamiltonian> hamiltonian, std::unique_ptr<Geometry> geometry, std::unique_ptr<Parameters_Method_LLG> llg_params, bool iteration_allowed);
		// Copy Constructor
		Spin_System(Spin_System const & other);
		// Assignment operator
		Spin_System& operator=(Spin_System const & other);

		// Update
		void UpdateEnergy();
		void UpdateEffectiveField();

		// Number of spins
		int nos;
		// Orientations of the Spins: spins[dim][nos]
		std::shared_ptr<std::vector<scalar>> spins;
		// Spin Hamiltonian
		std::shared_ptr<Engine::Hamiltonian> hamiltonian;
		// Geometric Information
		std::shared_ptr<Geometry> geometry;
		// Parameters for LLG Iterations (MC, SIB, ...)
		std::shared_ptr<Parameters_Method_LLG> llg_parameters;
		// Is it allowed to iterate on this system?
		bool iteration_allowed;

		// Total Energy of the spin system (to be updated from outside, i.e. SIB, GNEB, ...)
		scalar E;
		std::vector<scalar> E_array;
		// Total effective field of the spins [3][nos]
		std::vector<scalar> effective_field;


	//private:

	};
}
#endif