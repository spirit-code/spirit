#pragma once
#ifndef OPTIMIZER_NCG_H
#define OPTIMIZER_NCG_H

#include <engine/Optimizer.hpp>

namespace Engine
{
	/*
		Conjugate Gradient Optimizer
	*/
	class Optimizer_NCG : public Optimizer
	{
	public:
		Optimizer_NCG(std::shared_ptr<Engine::Method> method);

		// One Iteration
		void Iteration() override;

		// Optimizer name as string
		std::string Name() override;
		std::string FullName() override;
    };
}

#endif