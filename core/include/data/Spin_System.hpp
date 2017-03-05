#pragma once
#ifndef DATA_SPIN_SYSTEM_H
#define DATA_SPIN_SYSTEM_H

#include <random>
#include <memory>

#include "Spirit_Defines.h"
#include <engine/Vectormath_Defines.hpp>
#include <engine/Hamiltonian.hpp>
#include <engine/Hamiltonian_Isotropic.hpp>
#include <engine/Hamiltonian_Anisotropic.hpp>
#include <engine/Hamiltonian_Gaussian.hpp>
#include <data/Geometry.hpp>
#include <data/Parameters_Method_LLG.hpp>
#include <data/Parameters_Method_GNEB.hpp>

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

		// For multithreading
		void Lock() const;
		void Unlock() const;

		// Number of spins
		int nos;
		// Orientations of the Spins: spins[dim][nos]
		std::shared_ptr<vectorfield> spins;
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
		std::vector<std::pair<std::string, scalar>> E_array;
		// Mean of magnetization
		Vector3 M;
		// Total effective field of the spins [3][nos]
		vectorfield effective_field;

	private:
		// Mutex for thread-safety
		mutable std::mutex mutex;
	};
}
#endif