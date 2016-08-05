#pragma once
#ifndef DATA_PARAMETERS_LLG_H
#define DATA_PARAMETERS_LLG_H

#include <random>
#include <vector>

#include "Parameters_Method.h"

namespace Data
{
	// LLG_Parameters contains all LLG information about the spin system
	class Parameters_LLG : public Parameters_Method
	{
	public:
		// Constructor
		Parameters_LLG(std::string output_folder, int seed, int n_iterations, int log_steps, double temperature, double damping, double time_step, bool renorm_sd, double stt_magnitude, std::vector<double> stt_polarisation_normal, double force_convergence);

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

		// spin-transfer-torque parameter (prop to injected current density)
		double stt_magnitude;
		// spin_current polarisation normal vector
		std::vector<double> stt_polarisation_normal;
	};
}
#endif