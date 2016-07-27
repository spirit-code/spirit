#pragma once
#ifndef SOLVER_LLG_H
#define SOLVER_LLG_H

#include "Method.h"
#include "Spin_System.h"
#include "Spin_System_Chain.h"
#include "Optimizer.h"
#include "Optimizer_SIB.h"

#include <vector>

namespace Engine
{
	class Solver_LLG : public Method
	{
	public:
		// replace this depedency on the system chain with a vector<Spin_System>
        Solver_LLG(std::shared_ptr<Data::Spin_System_Chain> c, std::shared_ptr<Optimizer> optim);
        
        // Iteratively apply the GNEB method to the Spin System Chain
		// Output and console notification is generated every log_steps
		void Iterate() override;

		// Iterates the system one step with the semi-implicit midpoint solver method B
		void Iteration() override;

	private:
		// Save the current Step's Data: spins and energy
		void Save_Step(int image, int iteration, std::string suffix) override;
    };
}

#endif