#pragma once
#ifndef DATA_PARAMETERS_SOLVER_H
#define DATA_PARAMETERS_SOLVER_H


namespace Data
{
	// Solver Parameters Base Class
	class Parameters_Solver
	{
	public:
		// Renormalise after each iteration? -- maybe let the optimizer decide when to renormalize?
		//bool renorm = false;
		// Data output folder
		std::string output_folder;

		// Force convergence criterium
		double force_convergence;
	};
}
#endif