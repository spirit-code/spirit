#pragma once
#ifndef SOLVER_GNEB_H
#define SOLVER_GNEB_H

#include <vector>

#include "Solver.h"
#include "Optimizer_Heun.h"
#include <Spin_System_Chain.h>

namespace Engine
{
	class Solver_GNEB : public Solver
	{
	public:
        Solver_GNEB(std::shared_ptr<Data::Spin_System_Chain> c, std::shared_ptr<Optimizer> optimizer);
        
        // Iteratively apply the GNEB method to the Spin System Chain
		// Output and console notification is generated every log_steps
		void Iterate() override;

		// Iterates the system one step with the semi-implicit midpoint solver method B
		void Iteration() override;

	private:
		// Save the current Step's Data: images and images' energies and reaction coordinates
		void Save_Step(int image, int iteration, std::string suffix) override;
    };
}

#endif