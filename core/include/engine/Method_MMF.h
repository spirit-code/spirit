#pragma once
#ifndef METHOD_MMF_H
#define METHOD_MMF_H

#include "Method.h"

namespace Engine
{
	/*
		The Minimum Mode Following (MMF) method
	*/
	class Method_MMF : public Method
	{

	public:
		Method_MMF(std::shared_ptr<Data::Spin_System_Chain> c, std::shared_ptr<Optimizer> optim);

		// Iteratively apply the MMF method to the Spin System Chain
		// Output and console notification is generated every log_steps
		void Iterate();

		// Iterates the system one step
		void Iteration();

	};
}

#endif