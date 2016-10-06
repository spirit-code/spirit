#pragma once
#ifndef METHOD_MMF_H
#define METHOD_MMF_H

#include "Method.h"
#include "Parameters_Method_MMF.h"
#include "Spin_System_Chain_Collection.h"

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
		void Calculate_Force(std::vector<std::shared_ptr<std::vector<double>>> configurations, std::vector<std::vector<double>> & forces) override;
		
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

		std::vector<std::vector<double>> hessian;
		// Last calculated forces
		std::vector<std::vector<double>> F_gradient;
		// Last calculated minimum mode
		std::vector<std::vector<double>> minimum_mode;

		// Last iterations spins and reaction coordinate
		double Rx_last;
		std::vector<std::vector<double>> spins_last;

		// Which minimum mode function to use
		// ToDo: move into parameters
		std::string mm_function;

		// Functions for getting the minimum mode of a Hessian
		void Calculate_Force_Spectra_Matrix(std::vector<std::shared_ptr<std::vector<double>>> configurations, std::vector<std::vector<double>> & forces);
		void Calculate_Force_Spectra_Prefactor(std::vector<std::shared_ptr<std::vector<double>>> configurations, std::vector<std::vector<double>> & forces);
		void Calculate_Force_Lanczos(std::vector<std::shared_ptr<std::vector<double>>> configurations, std::vector<std::vector<double>> & forces);
	};
}

#endif