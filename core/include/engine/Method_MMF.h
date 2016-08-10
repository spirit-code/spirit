#pragma once
#ifndef METHOD_MMF_H
#define METHOD_MMF_H

#include "Method.h"
#include "Parameters_MMF.h"

namespace Engine
{
	/*
		The Minimum Mode Following (MMF) method
	*/
	class Method_MMF : public Method
	{

	public:
 		// Constructor
		Method_MMF(std::shared_ptr<Data::Parameters_MMF> parameters, int idx_img, int idx_chain);
    
	//public override:
		// Calculate Forces onto Systems
		void Calculate_Force(std::vector<std::shared_ptr<std::vector<double>>> configurations, std::vector<std::vector<double>> & forces) override;
		
		// Check if the Forces are converged
		bool Force_Converged() override;

		// Method name as string
		std::string Name() override;
	
		// Save the current Step's Data: images and images' energies and reaction coordinates
		void Save_Current(std::string starttime, int iteration, bool final) override;
		// A hook into the Optimizer before an Iteration
		void Hook_Pre_Iteration() override;
		// A hook into the Optimizer after an Iteration
		void Hook_Post_Iteration() override;
		
	private:
	};
}

#endif