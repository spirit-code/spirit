#pragma once
#ifndef METHOD_GNEB_H
#define METHOD_GNEB_H

#include <vector>

#include "Method.h"
#include <Spin_System_Chain.h>

namespace Engine
{
	/*
		The geodesic nudged elastic band (GNEB) method
	*/
	class Method_GNEB : public Method
	{
	public:
        // Constructor
		Method_GNEB(std::shared_ptr<Data::Spin_System_Chain> chain, int idx_img, int idx_chain);
    
		// Calculate Forces onto Systems
		void Calculate_Force(std::vector<std::shared_ptr<std::vector<double>>> configurations, std::vector<std::vector<double>> & forces) override;
		
		// Check if the Forces are converged
		bool Force_Converged() override;

		// Method name as string
		std::string Name() override;

	private:
		std::shared_ptr<Data::Spin_System_Chain> chain;
		// Save the current Step's Data: images and images' energies and reaction coordinates
		void Save_Step(std::string starttime, int iteration, bool final=false) override;
		// A hook into the Optimizer before an Iteration
		void Hook_Pre_Step() override;
		// A hook into the Optimizer after an Iteration
		void Hook_Post_Step() override;
    };
}

#endif