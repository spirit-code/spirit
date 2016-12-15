#pragma once
#ifndef DATA_PARAMETERS_METHOD_H
#define DATA_PARAMETERS_METHOD_H

#include "Core_Defines.h"

#include <string>

namespace Data
{
	// Solver Parameters Base Class
	class Parameters_Method
	{
	public:
		// Constructor
		Parameters_Method(std::string output_folder, scalar force_convergence, long int n_iterations, long int n_iterations_log);

		// Data output folder
		std::string output_folder;

		// Force convergence criterium
		scalar force_convergence;

		// Number of iterations carried out when pressing "play" or calling "iterate"
		long int n_iterations;
		// Number of iterations after which the Method should save data
		long int n_iterations_log;
	};
}
#endif