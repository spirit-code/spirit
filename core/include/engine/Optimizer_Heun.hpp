#pragma once
#ifndef OPTIMIZER_HEUN_H
#define OPTIMIZER_HEUN_H

#include <vector>

#include "Core_Defines.h"
#include "Vectormath_Defines.hpp"
#include "Optimizer.hpp"
#include "Spin_System_Chain.hpp"

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
		Optimizer_Heun(std::shared_ptr<Engine::Method> method);
		
		// One Iteration
		void Iteration() override;

		// Optimizer name as string
		std::string Name() override;
		std::string FullName() override;

    private:
		// Temporary Spins arrays
		std::vector<std::vector<Vector3>> spins_temp;

		// Virtual Heun Forces used in the Steps
		std::vector<std::vector<Vector3>> virtualforce;

		// TODO: THE HEUN METHOD CAN BE REWRITTEN TO BE NICER:
		//// Calculate the virtual Heun force to be used in the Steps
		//void VirtualForce(const int nos, std::vector<scalar> & spins, std::vector<scalar> & beff, scalar dt, std::vector<scalar> & force);
		//// First Part of one Optimization step
		//void FirstStep(const int nos, std::vector<scalar> & spins, scalar dt, std::vector<scalar> & force, std::vector<scalar> & spins_temp);
		//// Second Part of one Optimization step
		//void SecondStep(const int nos, std::vector<scalar> & spins, scalar dt, std::vector<scalar> & force);
    };
}

#endif