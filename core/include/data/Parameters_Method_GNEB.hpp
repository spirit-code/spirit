#pragma once
#ifndef DATA_PARAMETERS_METHOD_GNEB_H
#define DATA_PARAMETERS_METHOD_GNEB_H

#include <random>
#include <vector>

#include "Core_Defines.h"
#include <data/Parameters_Method.hpp>

namespace Data
{
	// LLG_Parameters contains all LLG information about the spin system
	class Parameters_Method_GNEB : public Parameters_Method
	{
	public:
		// Constructor
		Parameters_Method_GNEB(std::string output_folder, scalar force_convergence, long int n_iterations, long int n_iterations_log, scalar spring_constant, int n_E_interpolations);

		// Strength of springs between images
		scalar spring_constant;

		// Number of Energy interpolations between Images
		int n_E_interpolations;
	};
}
#endif