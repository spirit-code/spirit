#pragma once
#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "Spirit_Defines.h"
#include <engine/Method.hpp>
#include <utility/Logging.hpp>

#include <vector>
#include <algorithm>
#include <iterator>
#include <string>



namespace Engine
{
	/*
		An optimizer is iterated on a set of Spin Systems under applied force:
			Each Step of the optimizer should move the systems according to the applied force.
			Note that the applied force may need to be recalculated during one Step
	*/
	class Optimizer
	{
	public:
		// The Optimizer needs to be configured by the Solver after creation
		Optimizer(std::shared_ptr<Engine::Method> method);

		// One Iteration
		virtual void Iteration();

		// Iterate for method->parameters->n iterations
		virtual void Iterate() final;

		// Calculate a smooth but current IPS value
		virtual scalar getIterationsPerSecond() final;

		// Optimizer name as string
		virtual std::string Name();
		virtual std::string FullName();

	protected:
		// The Method instance with which to calculate the forces on configurations
		std::shared_ptr<Engine::Method> method;

		// Number of Images
		int noi;
		// Number of Spins
		int nos;
		// Number of iterations
		int n_iterations;
		// Number of iterations until log
		int n_iterations_log;
		// Number of times to save
		int n_log;
    // Number of iterations that have been executed
    int iteration;
    // Number of steps (set of iterations between logs) that have been executed
    int step;

		// Pointers to Configurations
		std::vector<std::shared_ptr<vectorfield>> configurations;
		// Actual Forces on the configurations
		std::vector<vectorfield> force;

		// The time at which this Solver's Iterate() was last called
		std::string starttime;
		// Timings and Iterations per Second
		scalar ips;
		std::deque<std::chrono::time_point<std::chrono::system_clock>> t_iterations;

		// Check if a stop file is present -> Stop the iterations
		virtual bool StopFilePresent() final;

		// Precision for the conversion of scalar to string
		int print_precision;
	};
}
#endif