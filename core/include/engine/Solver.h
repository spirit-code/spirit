#pragma once
#ifndef SOLVER_H
#define SOLVER_H

#include <fstream>

#include "Spin_System_Chain.h"
#include "Optimizer.h"
#include "Optimizer_SIB.h"
#include "Optimizer_SIB2.h"
#include "Optimizer_Heun.h"
#include "Optimizer_CG.h"
#include "Optimizer_QM.h"
#include "Timing.h"

namespace Engine
{
	/*
		Base Class for Solvers
	*/
	class Solver
	{
	public:
		// Constructor to be used in derived classes
		Solver(std::shared_ptr<Data::Spin_System_Chain> c, std::shared_ptr<Optimizer> optimizer)
		{
			this->c = c;
			this->optimizer = optimizer;
			this->starttime = Utility::Timing::CurrentDateTime();
		}

		// Iterate for n iterations
		virtual void Iterate(int n_iterations, int log_steps)
		{
			// Not Implemented!
			Utility::Log.Send(Utility::Log_Level::L_ERROR, Utility::Log_Sender::ALL, std::string("Tried to use Solver::Iterate() of the Solver base class!"));
			throw Utility::Exception::Not_Implemented;
		}

		// One Iteration
		virtual void Iteration()
		{
			// Not Implemented!
			Utility::Log.Send(Utility::Log_Level::L_ERROR, Utility::Log_Sender::ALL, std::string("Tried to use Solver::Iteration() of the Solver base class!"));
			throw Utility::Exception::Not_Implemented;
		}

	protected:
		// The Image Chain on which this Solver operates
		std::shared_ptr<Data::Spin_System_Chain> c;

		// The Images to operate on
		std::vector<std::shared_ptr<Data::Spin_System>> systems;

		// The Force corresponding to this solver
		std::shared_ptr<Engine::Force> force_call;

		// Optimizer to iterate on the image(s)
		std::shared_ptr<Optimizer> optimizer;

		// The time at which this Solver's Iterate() was last called
		std::string starttime;

		// Save the current Step's Data
		virtual void Save_Step(int image, int iteration, std::string suffix)
		{
			// Not Implemented!
			Utility::Log.Send(Utility::Log_Level::L_ERROR, Utility::Log_Sender::ALL, std::string("Tried to use Solver::Save_Step() of the Solver base class!"));
			throw Utility::Exception::Not_Implemented;
		}

		//// Create the Force specific to the Solver
		//virtual void Configure()
		//{
		//	this->force_call = std::shared_ptr<Force>(new Force(c));
		//	// Not Implemented!
		//	Utility::Log.Send(Utility::Log_Level::L_ERROR, Utility::Log_Sender::ALL, std::string("Tried to use Solver::Configure() of the Solver base class!"));
		//	//throw Utility::Exception::Not_Implemented;
		//}

		// Check if a stop file is present -> Stop the iterations
		bool StopFilePresent()
		{
			std::ifstream f("STOP");
			return f.good();
		}
	};
}

#endif