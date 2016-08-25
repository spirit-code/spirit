#pragma once
#ifndef DATA_PARAMETERS_MMF_H
#define DATA_PARAMETERS_MMF_H

#include <random>
#include <vector>

#include "Parameters_Method.h"

namespace Data
{
	// LLG_Parameters contains all LLG information about the spin system
	class Parameters_MMF : public Parameters_Method
	{
	public:
		Parameters_MMF(std::string output_folder, double force_convergence, int n_iterations, int log_steps);
	};
}
#endif