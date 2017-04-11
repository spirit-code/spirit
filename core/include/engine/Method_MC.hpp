#pragma once
#ifndef METHOD_MC_H
#define METHOD_MC_H

#include "Spirit_Defines.h"
#include <engine/Method.hpp>
#include <data/Spin_System.hpp>
// #include <data/Parameters_Method_MC.hpp>

#include <vector>

namespace Engine
{
	/*
		The Monte Carlo method
	*/
	class Method_MC : public Method
	{
	public:
        // Constructor
		Method_MC(std::shared_ptr<Data::Spin_System> system, int idx_img, int idx_chain);
    
		// Calculate Forces onto Systems
		void Calculate_Force(std::vector<std::shared_ptr<vectorfield>> configurations, std::vector<vectorfield> & forces) override;
		
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
    };
}

#endif