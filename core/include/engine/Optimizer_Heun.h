#pragma once
#ifndef OPTIMIZER_HEUN_H
#define OPTIMIZER_HEUN_H

#include <vector>

#include "Optimizer.h"
#include "Spin_System_Chain.h"

namespace Engine
{
	/*
		The Heun method is a direct optimization of a Spin System:
			The Spin System will follow the applied force on a direct trajectory, which need not be physical.
			See also https://en.wikipedia.org/wiki/Heun%27s_method
	*/
	class Optimizer_Heun : public Optimizer
	{

	public:
		Optimizer_Heun(std::vector<std::shared_ptr<Data::Spin_System>> systems, std::shared_ptr<Engine::Method> method);
		
		// One Iteration
		void Iteration() override;

		// Optimizer name as string
		std::string Name() override;
		std::string FullName() override;

    private:
		// Temporary Spins arrays
		std::vector<std::vector<double>> spins_temp;

		// Virtual Heun Forces used in the Steps
		std::vector<std::vector<double>> virtualforce;

		// TODO: THE HEUN METHOD CAN BE REWRITTEN TO BE NICER:
		//// Calculate the virtual Heun force to be used in the Steps
		//void VirtualForce(const int nos, std::vector<double> & spins, std::vector<double> & beff, double dt, std::vector<double> & force);
		//// First Part of one Optimization step
		//void FirstStep(const int nos, std::vector<double> & spins, double dt, std::vector<double> & force, std::vector<double> & spins_temp);
		//// Second Part of one Optimization step
		//void SecondStep(const int nos, std::vector<double> & spins, double dt, std::vector<double> & force);
    };
}

#endif