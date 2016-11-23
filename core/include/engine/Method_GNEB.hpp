#pragma once
#ifndef METHOD_GNEB_H
#define METHOD_GNEB_H

#include <vector>

#include "Core_Defines.h"
#include "Method.hpp"
#include <Spin_System_Chain.hpp>

namespace Engine
{
	/*
		The geodesic nudged elastic band (GNEB) method
	*/
	class Method_GNEB : public Method
	{
	public:
        // Constructor
		Method_GNEB(std::shared_ptr<Data::Spin_System_Chain> chain, int idx_chain);
    
		// Calculate Forces onto Systems
		void Calculate_Force(std::vector<std::shared_ptr<std::vector<Vector3>>> configurations, std::vector<std::vector<Vector3>> & forces) override;
		
		// Check if the Forces are converged
		bool Force_Converged() override;

		// Method name as string
		std::string Name() override;

		// Save the current Step's Data: images and images' energies and reaction coordinates
		void Save_Current(std::string starttime, int iteration, bool initial=false, bool final=false) override;
		// A hook into the Optimizer before an Iteration
		void Hook_Pre_Iteration() override;
		// A hook into the Optimizer after an Iteration
		void Hook_Post_Iteration() override;

		// Sets iteration_allowed to false for the chain
		void Finalize() override;
		
		bool Iterations_Allowed() override;

	private:
		std::shared_ptr<Data::Spin_System_Chain> chain;

		// Last calculated energies
		std::vector<scalar> energies;
		// Last calculated Reaction coordinates
		std::vector<scalar> Rx;
		// Last calculated forces
		std::vector<std::vector<Vector3>> F_total;
		std::vector<std::vector<Vector3>> F_gradient;
		std::vector<std::vector<Vector3>> F_spring;
		// Last calculated tangents
		std::vector<std::vector<Vector3>> tangents;
    };
}

#endif