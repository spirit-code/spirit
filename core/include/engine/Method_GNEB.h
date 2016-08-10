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

		// Save the current Step's Data: images and images' energies and reaction coordinates
		void Save_Current(std::string starttime, int iteration, bool final=false) override;
		// A hook into the Optimizer before an Iteration
		void Hook_Pre_Iteration() override;
		// A hook into the Optimizer after an Iteration
		void Hook_Post_Iteration() override;

		// Sets iteration_allowed to false for the corresponding method
		void Finalize() override;
		
		bool Iterations_Allowed() override;
	private:
		std::shared_ptr<Data::Spin_System_Chain> chain;

		// Last calculated energies
		std::vector<double> energies;
		// Last calculated Reaction coordinates
		std::vector<double> Rx;
		// Last calculated forces
		std::vector<std::vector<double>> F_total;
		std::vector<std::vector<double>> F_gradient;
		std::vector<std::vector<double>> F_spring;
		// Last calculated tangents
		std::vector<std::vector<double>> tangents;
    };
}

#endif