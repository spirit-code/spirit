#pragma once
#ifndef OPTIMIZER_SIB2_H
#define OPTIMIZER_SIB2_H

#include <vector>

#include "Optimizer.h"
#include "Spin_System_Chain.h"
#include "Force.h"

namespace Engine
{
	/*
		Nikolai's version of the SIB Optimizer... Translated from his code... May be buggy, but so might the original SIB...
	*/
	class Optimizer_SIB2 : public Optimizer
	{

	public:
		// One step in the optimization
		void Step() override;
		void Configure(std::vector<std::shared_ptr<Data::Spin_System>> systems, std::shared_ptr<Engine::Force> force_call) override;
		// Optimizer name as string
		std::string Name() override;
		std::string Fullname() override;
    private:

		// Temporary Spins arrays
		std::vector<std::vector<double>> spins_temp;

		void Gen_Xi(Data::Spin_System & s, std::vector<double> & xi, double eps);
    };
}

#endif