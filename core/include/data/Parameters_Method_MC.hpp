#pragma once
#ifndef DATA_PARAMETERS_METHOD_MC_H
#define DATA_PARAMETERS_METHOD_MC_H

#include <random>
#include <vector>

#include "Spirit_Defines.h"
#include <data/Parameters_Method.hpp>

namespace Data
{
	// LLG_Parameters contains all LLG information about the spin system
	class Parameters_Method_MC : public Parameters_Method
	{
	public:
		Parameters_Method_MC(std::string output_folder, std::array<bool,10> output, long int n_iterations, long int n_iterations_log,
			long int max_walltime_sec, int seed_i, scalar temperature_i, scalar acceptance_ratio_i);

		//PRNG Seed
		const int seed;
		// --------------- Different distributions ------------
		std::mt19937 prng;
		std::uniform_real_distribution<scalar> distribution_real;
		std::uniform_real_distribution<scalar> distribution_minus_plus_one;
		std::uniform_int_distribution<int> distribution_int;

		// Temperature [K]
		scalar temperature;

		// Step acceptance ratio
		scalar acceptance_ratio;

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