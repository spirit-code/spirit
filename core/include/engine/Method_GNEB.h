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
		Method_GNEB(std::shared_ptr<Data::Parameters_GNEB> parameters);
    
	//public override:
		// Calculate Forces onto Systems
		void Calculate_Force(std::vector<std::vector<double>> configurations, std::vector<std::vector<double>> & forces) override;
		
		// Check if the Forces are converged
		bool Force_Converged() override;

		// Method name as string
		std::string Name() override;

	private:
		// Save the current Step's Data: images and images' energies and reaction coordinates
		void Save_Step(int image, int iteration, std::string suffix) override;
		// A hook into the Optimizer before an Iteration
		void Hook_Pre_Step() override;
		// A hook into the Optimizer after an Iteration
		void Hook_Post_Step() override;
    };
}

#endif