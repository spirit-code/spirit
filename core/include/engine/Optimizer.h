#pragma once
#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "Spin_System.h"
#include "Force.h"
#include <vector>
#include <algorithm>
#include <iterator>
#include <string>

#include "Logging.h"


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
		virtual void Configure(std::vector<std::shared_ptr<Data::Spin_System>> systems, std::shared_ptr<Engine::Force> force_call);

		// One step in the optimization
		virtual void Step();

		// Optimizer name as string
		virtual std::string Name();
		virtual std::string Fullname();

	protected:
		// The Spin Systems which to optimize
		std::vector<std::shared_ptr<Data::Spin_System>> systems;
		// The Force instance with which to calculate the forces on configurations
		std::shared_ptr<Engine::Force> force_call;

		// Number of Images
		int noi;
		// Number of Spins
		int nos;

		// The actual configurations of the Spin Systems
		std::vector<std::vector<double>> configurations;
		// Actual Forces on the configurations
		std::vector<std::vector<double>> force;
	};
}
#endif