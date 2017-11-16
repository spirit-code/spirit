#pragma once
#ifndef DATA_PARAMETERS_METHOD_LLG_H
#define DATA_PARAMETERS_METHOD_LLG_H

#include <vector>

#include "Spirit_Defines.h"
#include <engine/Vectormath_Defines.hpp>
#include <data/Parameters_Method_Solver.hpp>

namespace Data
{
	// LLG_Parameters contains all LLG information about the spin system
	class Parameters_Method_LLG : public Parameters_Method_Solver
	{
	public:
		// Constructor
		Parameters_Method_LLG(std::string output_folder, std::string output_file_tag, 
            std::array<bool,9> output, scalar force_convergence, long int n_iterations, 
            long int n_iterations_log, long int max_walltime_sec, std::shared_ptr<Pinning> pinning, 
            int rng_seed, scalar temperature, scalar damping, scalar beta, scalar time_step, 
            bool renorm_sd, bool stt_use_gradient, scalar stt_magnitude, 
            Vector3 stt_polarisation_normal);

		// Damping
		scalar damping;
		scalar beta;

		// Temperature [K]
		scalar temperature;
		// Seed for RNG
		int rng_seed;
		// Mersenne twister PRNG
		std::mt19937 prng;

		// - true:  use gradient approximation for STT
		// - false: use pinned monolayer approximation with current in z-direction
		bool stt_use_gradient;
		// Spin transfer torque parameter (prop to injected current density)
		scalar stt_magnitude;
		// Spin current polarisation normal vector
		Vector3 stt_polarisation_normal;

		// Do direct minimization instead of dynamics
		bool direct_minimization;


		// ----------------- Output --------------
		// Energy output settings
		bool output_energy_step;
		bool output_energy_archive;
		bool output_energy_spin_resolved;
		bool output_energy_divide_by_nspins;
		// Spin configurations output settings
		bool output_configuration_step;
		bool output_configuration_archive;
	};
}
#endif