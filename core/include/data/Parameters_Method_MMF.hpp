#pragma once
#ifndef DATA_PARAMETERS_METHOD_MMF_H
#define DATA_PARAMETERS_METHOD_MMF_H

#include <random>
#include <vector>

#include "Parameters_Method.hpp"

namespace Data
{
	// LLG_Parameters contains all LLG information about the spin system
	class Parameters_Method_MMF : public Parameters_Method
	{
	public:
		Parameters_Method_MMF(std::string output_folder, double force_convergence, long int n_iterations, long int n_iterations_log);
	};
}
#endif