#pragma once
#ifndef METHOD_LLG_H
#define METHOD_LLG_H

#include "Method.h"
#include "Parameters_LLG.h"
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
        // Constructor
		Method_LLG(std::shared_ptr<Data::Parameters_LLG> parameters);
    
	//public override:
		// Calculate Forces onto Systems
		void Calculate_Force(std::vector<std::vector<double>> configurations, std::vector<std::vector<double>> & forces) override;
		
		// Check if the Forces are converged
		bool Force_Converged() override;

		// Method name as string
		std::string Name() override;

	private:
		// Save the current Step's Data: spins and energy
		void Save_Step(int image, int iteration, std::string suffix) override;
		// A hook into the Optimizer before an Iteration
		void Hook_Pre_Step() override;
		// A hook into the Optimizer after an Iteration
		void Hook_Post_Step() override;
    };
}

#endif