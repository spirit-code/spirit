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
		virtual void Configure(std::vector<std::shared_ptr<Data::Spin_System>> systems, std::shared_ptr<Engine::Force> force_call)
		{
			this->systems = systems;

			this->noi = systems.size();
			this->nos = systems[0]->nos;

			this->configurations = std::vector<std::vector<double>>(this->noi, std::vector<double>(3 * this->nos));
			for (int i = 0; i < this->noi; ++i)
			{
				this->configurations[i] = systems[i]->spins;
			}

			this->force = std::vector<std::vector<double>>(this->noi, std::vector<double>(3 * this->nos, 0));	// [noi][3*nos]

			this->force_call = force_call;
			// Calculate forces once, so that the Solver does not think it's converged
			this->force_call->Calculate(this->configurations, this->force);
		}

		// One step in the optimization
		virtual void Step()
		{
			// Not Implemented!
			Utility::Log.Send(Utility::Log_Level::L_ERROR, Utility::Log_Sender::ALL, std::string("Tried to use Optimizer::Step() of the Optimizer base class!"));
		}


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