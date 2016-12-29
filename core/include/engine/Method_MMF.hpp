#pragma once
#ifndef METHOD_MMF_H
#define METHOD_MMF_H

#include "Core_Defines.h"
#include <engine/Method.hpp>
#include <data/Parameters_Method_MMF.hpp>
#include <data/Spin_System_Chain_Collection.hpp>

namespace Engine
{
	/*
		The Minimum Mode Following (MMF) method
	*/
	class Method_MMF : public Method
	{
	public:
 		// Constructor
		Method_MMF(std::shared_ptr<Data::Spin_System_Chain_Collection> collection, int idx_chain);
    
	//public override:
		// Calculate Forces onto Systems
		void Calculate_Force(std::vector<std::shared_ptr<vectorfield>> configurations, std::vector<vectorfield> & forces) override;
		
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

		// Sets iteration_allowed to false for the collection
		void Finalize() override;
		
		bool Iterations_Allowed() override;
		
	private:
		bool switched1, switched2;
		std::shared_ptr<Data::Spin_System_Chain_Collection> collection;

		std::vector<MatrixX> hessian;
		// Last calculated forces
		std::vector<vectorfield> F_gradient;
		// Last calculated minimum mode
		std::vector<vectorfield> minimum_mode;

		// Last iterations spins and reaction coordinate
		scalar Rx_last;
		std::vector<vectorfield> spins_last;

		// Which minimum mode function to use
		// ToDo: move into parameters
		std::string mm_function;

		// Functions for getting the minimum mode of a Hessian
		void Calculate_Force_Spectra_Matrix(std::vector<std::shared_ptr<vectorfield>> configurations, std::vector<vectorfield> & forces);
	};
}

#endif