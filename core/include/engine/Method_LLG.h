#pragma once
#ifndef METHOD_LLG_H
#define METHOD_LLG_H

#include "Method.h"
#include "Parameters_Method_LLG.h"
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
		Method_LLG(std::shared_ptr<Data::Spin_System> system, int idx_img, int idx_chain);
    
		// Calculate Forces onto Systems
		void Calculate_Force(std::vector<std::shared_ptr<std::vector<double>>> configurations, std::vector<std::vector<double>> & forces) override;
		
		// Check if the Forces are converged
		bool Force_Converged() override;

		// Method name as string
		std::string Name() override;

		// Save the current Step's Data: spins and energy
		void Save_Current(std::string starttime, int iteration, bool initial=false, bool final=false) override;
		// A hook into the Optimizer before an Iteration
		void Hook_Pre_Iteration() override;
		// A hook into the Optimizer after an Iteration
		void Hook_Post_Iteration() override;

		// Sets iteration_allowed to false for the corresponding method
		void Finalize() override;

	private:
		// Last calculated forces
		std::vector<std::vector<double>> F_total;
		// Convergence parameters
		std::vector<bool> force_converged;
    };
}

#endif