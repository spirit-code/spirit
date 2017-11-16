#pragma once
#ifndef DATA_PARAMETERS_METHOD_SOLVER_H
#define DATA_PARAMETERS_METHOD_SOLVER_H

#include "Spirit_Defines.h"
#include <data/Parameters_Method.hpp>
#include <data/Pinning.hpp>

#include <string>
#include <array>
#include <random>

namespace Data
{
	// Solver Parameters Base Class
	class Parameters_Method_Solver : public Parameters_Method
	{
	public:
		// Constructor
		Parameters_Method_Solver(std::string output_folder, std::string output_file_tag, 
            std::array<bool,3> output, long int n_iterations, long int n_iterations_log,
			long int max_walltime_sec, std::shared_ptr<Pinning> pinning, scalar force_convergence,
			scalar dt);

		// Time step per iteration [ps]
		scalar dt;
	};
}
#endif