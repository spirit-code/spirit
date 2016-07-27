#pragma once
#ifndef METHOD_H
#define METHOD_H

#include <fstream>

#include "Spin_System_Chain.h"
#include "Optimizer.h"
#include "Optimizer_SIB.h"
#include "Optimizer_SIB2.h"
#include "Optimizer_Heun.h"
#include "Optimizer_CG.h"
#include "Optimizer_QM.h"
#include "Timing.h"

#include <deque>

namespace Engine
{
	/*
		Base Class for Methods
	*/
	class Method
	{
	public:
		// Constructor to be used in derived classes
		Method(std::shared_ptr<Data::Spin_System_Chain> c, std::shared_ptr<Optimizer> optimizer);

		// Iterate for n iterations
		virtual void Iterate();

		// One Iteration
		virtual void Iteration();

		double getIterationsPerSecond();

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

		// Timings and Iterations per Second
		double ips;
		std::deque<std::chrono::time_point<std::chrono::system_clock>> t_iterations;

		// Save the current Step's Data
		virtual void Save_Step(int image, int iteration, std::string suffix);

		//// Create the Force specific to the Solver
		//virtual void Configure()
		//{
		//	this->force_call = std::shared_ptr<Force>(new Force(c));
		//	// Not Implemented!
		//	Utility::Log.Send(Utility::Log_Level::L_ERROR, Utility::Log_Sender::ALL, std::string("Tried to use Solver::Configure() of the Solver base class!"));
		//	//throw Utility::Exception::Not_Implemented;
		//}

		// Check if a stop file is present -> Stop the iterations
		bool StopFilePresent();
	};
}

#endif