#pragma once
#ifndef Optimizer_VP_H
#define Optimizer_VP_H

#include "Optimizer.h"

namespace Engine
{
	/*
		Velocity Projection Optimizer
	*/
	class Optimizer_VP : public Optimizer
	{
	public:
		Optimizer_VP(std::shared_ptr<Engine::Method> method);

		void Iteration() override;
		
		// Optimizer name as string
		std::string Name() override;
		std::string FullName() override;

	private:
		// "Mass of our particle" which we accelerate
		double m = 1.0;

		// Temporary Spins arrays
		std::vector<std::vector<double>> spins_temp;
		// Force in previous step [noi][3nos]
		std::vector<std::vector<double>> force_previous;
		// Velocity in previous step [noi][3nos]
		std::vector<std::vector<double>> velocity_previous;
		// Velocity used in the Steps [noi][3nos]
		std::vector<std::vector<double>> velocity;
		// Projection of velocities onto the forces [noi]
		std::vector<double> projection;
		// |force|^2
		std::vector<double> force_norm2;
    };
}

#endif