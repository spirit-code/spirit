#pragma once
#ifndef OPTIMIZER_CG_H
#define OPTIMIZER_CG_H

#include "Optimizer.h"

namespace Engine
{
	/*
		Conjugate Gradient Optimizer
	*/
	class Optimizer_CG : public Optimizer
	{
	public:
		void Step() override {};

		// Optimizer name as string
		std::string Name() override;
		std::string Fullname() override;
    };
}

#endif