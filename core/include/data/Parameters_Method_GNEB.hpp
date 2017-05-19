#pragma once
#ifndef DATA_PARAMETERS_METHOD_GNEB_H
#define DATA_PARAMETERS_METHOD_GNEB_H

#include <random>
#include <vector>

#include "Spirit_Defines.h"
#include <data/Parameters_Method.hpp>

namespace Data
{
	// LLG_Parameters contains all LLG information about the spin system
	class Parameters_Method_GNEB : public Parameters_Method
	{
	public:
		// Constructor
		Parameters_Method_GNEB(std::string output_folder, std::array<bool,8> output, scalar force_convergence, long int n_iterations, long int n_iterations_log,
			long int max_walltime_sec, scalar spring_constant, int n_E_interpolations);

		bool output_energies_step;
		bool output_energies_divide_by_nspins;
		bool output_energies_interpolated;
		bool output_chain_step;
		
		// Strength of springs between images
		scalar spring_constant;

		// Number of Energy interpolations between Images
		int n_E_interpolations;
	};
}
#endif