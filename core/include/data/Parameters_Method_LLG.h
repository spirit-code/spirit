#pragma once
#ifndef DATA_PARAMETERS_METHOD_LLG_H
#define DATA_PARAMETERS_METHOD_LLG_H

#include <random>
#include <vector>

#include "Parameters_Method.h"

namespace Data
{
	// LLG_Parameters contains all LLG information about the spin system
	class Parameters_Method_LLG : public Parameters_Method
	{
	public:
		// Constructor
		Parameters_Method_LLG(std::string output_folder, double force_convergence, long int n_iterations, long int n_iterations_log,
			int seed_i, double temperature_i, double damping_i, double time_step_i,
			bool renorm_sd_i, bool save_single_configurations_i,
			double stt_magnitude_i, std::vector<double> stt_polarisation_normal_i);

		//PRNG Seed
		const int seed;
		// --------------- Different distributions ------------
		std::mt19937 prng;
		std::uniform_real_distribution<double> distribution_real;
		std::uniform_real_distribution<double> distribution_minus_plus_one;
		std::uniform_int_distribution<int> distribution_int;

		// Temperature [K]
		double temperature;
		// Damping
		double damping;
		// Time step per iteration
		double dt;
		// whether to renormalize spins after every SD iteration
		bool renorm_sd = 1;
		// Whether to save a single "spins"
		bool save_single_configurations;

		// spin-transfer-torque parameter (prop to injected current density)
		double stt_magnitude;
		// spin_current polarisation normal vector
		std::vector<double> stt_polarisation_normal;
	};
}
#endif