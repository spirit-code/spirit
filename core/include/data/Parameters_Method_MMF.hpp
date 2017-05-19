#pragma once
#ifndef DATA_PARAMETERS_METHOD_MMF_H
#define DATA_PARAMETERS_METHOD_MMF_H

#include <random>
#include <vector>

#include "Spirit_Defines.h"
#include <data/Parameters_Method.hpp>

namespace Data
{
	// LLG_Parameters contains all LLG information about the spin system
	class Parameters_Method_MMF : public Parameters_Method
	{
	public:
		Parameters_Method_MMF(std::string output_folder, std::array<bool,9> output, scalar force_convergence, long int n_iterations, long int n_iterations_log, long int max_walltime_sec);

		// Energy output settings
		bool output_energy_step;
		bool output_energy_archive;
		bool output_energy_divide_by_nspins;

		// Spin configurations output settings
		bool output_configuration_step;
		bool output_configuration_archive;
	};
}
#endif