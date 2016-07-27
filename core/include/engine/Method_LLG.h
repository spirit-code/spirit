#pragma once
#ifndef METHOD_LLG_H
#define METHOD_LLG_H

#include "Method.h"
#include "Spin_System.h"
#include "Spin_System_Chain.h"
#include "Optimizer.h"
#include "Optimizer_SIB.h"

#include <vector>

namespace Engine
{
	/*
		The Landau-Lifshitz-Gilbert (LLG) method
	*/
	class Method_LLG : public Method
	{
	public:
		// replace this depedency on the system chain with a vector<Spin_System>
        Method_LLG(std::shared_ptr<Data::Spin_System_Chain> c, std::shared_ptr<Optimizer> optim);
        
        // Iteratively apply the GNEB method to the Spin System Chain
		// Output and console notification is generated every log_steps
		void Iterate() override;

		// Iterates the system one step
		void Iteration() override;

	private:
		// Save the current Step's Data: spins and energy
		void Save_Step(int image, int iteration, std::string suffix) override;
    };
}

#endif