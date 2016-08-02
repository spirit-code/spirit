#pragma once
#ifndef DATA_PARAMETERS_METHOD_H
#define DATA_PARAMETERS_METHOD_H

#include <string>

namespace Data
{
	// Solver Parameters Base Class
	class Parameters_Method
	{
	public:
		// number of iterations carried out when pressing "play" or calling "iterate"
		int n_iterations;
		// after "log_steps"-iterations the current system is logged to file
		int log_steps;

		// Renormalise after each iteration? -- maybe let the optimizer decide when to renormalize?
		//bool renorm = false;
		// Data output folder
		std::string output_folder;

		// Force convergence criterium
		double force_convergence;
	};
}
#endif