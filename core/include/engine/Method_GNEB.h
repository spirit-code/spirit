#pragma once
#ifndef METHOD_GNEB_H
#define METHOD_GNEB_H

#include <vector>

#include "Method.h"
#include "Optimizer_Heun.h"
#include <Spin_System_Chain.h>

namespace Engine
{
	/*
		The geodesic nudged elastic band (GNEB) method
	*/
	class Method_GNEB : public Method
	{
	public:
        Method_GNEB(std::shared_ptr<Data::Spin_System_Chain> c, std::shared_ptr<Optimizer> optimizer);
        
        // Iteratively apply the GNEB method to the Spin System Chain
		// Output and console notification is generated every log_steps
		void Iterate() override;

		// Iterates the system one step
		void Iteration() override;

		// Optimizer name as string
		std::string Name() override;

	private:
		// Save the current Step's Data: images and images' energies and reaction coordinates
		void Save_Step(int image, int iteration, std::string suffix) override;
    };
}

#endif