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
		void Step() override {};
    };
}

#endif