#pragma once
#ifndef OPTIMIZER_QM_H
#define OPTIMIZER_QM_H

#include "Optimizer.h"

namespace Engine
{
	/*
		Quick-Min Optimizer
	*/
	class Optimizer_QM : public Optimizer
	{
	public:
		Optimizer_QM(std::shared_ptr<Engine::Method> method);

		void Iteration() override;
		
		// Optimizer name as string
		std::string Name() override;
		std::string FullName() override;
    };
}

#endif