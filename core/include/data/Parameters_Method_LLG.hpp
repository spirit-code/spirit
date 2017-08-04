#pragma once
#ifndef DATA_PARAMETERS_METHOD_LLG_H
#define DATA_PARAMETERS_METHOD_LLG_H

#include <vector>

#include "Spirit_Defines.h"
#include <engine/Vectormath_Defines.hpp>
#include <data/Parameters_Method.hpp>

namespace Data
{
	// LLG_Parameters contains all LLG information about the spin system
	class Parameters_Method_LLG : public Parameters_Method
	{
	public:
		// Constructor
		Parameters_Method_LLG(std::string output_folder, std::array<bool,10> output, scalar force_convergence, long int n_iterations, long int n_iterations_log,
			long int max_walltime_sec, std::shared_ptr<Pinning> pinning, int rng_seed, scalar temperature_i, scalar damping_i, scalar time_step_i, bool renorm_sd_i,
			scalar stt_magnitude_i, Vector3 stt_polarisation_normal_i);

		// Temperature [K]
		scalar temperature;
		// Damping
		scalar damping;
		// Time step per iteration
		scalar dt;
		// whether to renormalize spins after every SD iteration
		bool renorm_sd = 1;

		// Energy output settings
		bool output_energy_step;
		bool output_energy_archive;
		bool output_energy_spin_resolved;
		bool output_energy_divide_by_nspins;

		// Spin configurations output settings
		bool output_configuration_step;
		bool output_configuration_archive;

		// spin-transfer-torque parameter (prop to injected current density)
		scalar stt_magnitude;
		// spin_current polarisation normal vector
		Vector3 stt_polarisation_normal;
	};
}
#endif