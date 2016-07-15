#pragma once
#ifndef DATA_PARAMETERS_GNEB_H
#define DATA_PARAMETERS_GNEB_H

#include <random>
#include <vector>

#include "Parameters_Solver.h"

namespace Data
{
	// LLG_Parameters contains all LLG information about the spin system
	class Parameters_GNEB : public Parameters_Solver
	{
	public:
		// Constructor
		Parameters_GNEB(std::string output_folder, double spring_constant, double force_convergence, int n_iterations, int log_steps, int n_E_interpolations);

		// Strength of springs between images
		double spring_constant;
		
		// number of iterations carried out when pressing "play" or calling "iterate"
		int n_iterations;
		// after "log_steps"-iterations the current system is logged to file
		int log_steps;

		// Number of Energy interpolations between Images
		int n_E_interpolations;
	};
}
#endif