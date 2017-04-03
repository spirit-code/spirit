#pragma once
#ifndef DATA_PARAMETERS_METHOD_H
#define DATA_PARAMETERS_METHOD_H

#include "Spirit_Defines.h"

#include <string>
#include <array>

namespace Data
{
	// Solver Parameters Base Class
	class Parameters_Method
	{
	public:
		// Constructor
		Parameters_Method(std::string output_folder, std::array<bool,3> output, scalar force_convergence, long int n_iterations, long int n_iterations_log);

		// Data output folder
		std::string output_folder;

		// Save any output when logging
		bool output_any;
		// Save output at initial state
		bool output_initial;
		// Save output at final state
		bool output_final;

		// Force convergence criterium
		scalar force_convergence;

		// Number of iterations carried out when pressing "play" or calling "iterate"
		long int n_iterations;
		// Number of iterations after which the Method should save data
		long int n_iterations_log;
	};
}
#endif