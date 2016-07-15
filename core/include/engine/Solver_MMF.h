#pragma once
#ifndef SOLVER_MMF_H
#define SOLVER_MMF_H

#include "Solver.h"

namespace Engine
{
	/*
		Solver for the Minimum Mode Following (MMF) method
	*/
	class Solver_MMF : public Solver
	{

	public:
		Solver_MMF(std::shared_ptr<Data::Spin_System_Chain> c, std::shared_ptr<Optimizer> optim);

		// Iteratively apply the MMF method to the Spin System Chain
		// Output and console notification is generated every log_steps
		void Iterate();

		// Iterates the system one step with the semi-implicit midpoint solver method B
		void Iteration();

	};
}

#endif