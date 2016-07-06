#pragma once
#ifndef FORCE_LLG_H
#define FORCE_LLG_H

#include "Force.h"

namespace Engine
{
	class Force_LLG : public Force
	{
	public:
		Force_LLG(std::shared_ptr<Data::Spin_System_Chain> c);
		void Calculate(std::vector<std::vector<double>> & configurations, std::vector<std::vector<double>> & forces) override;

		bool IsConverged() override;

	private:
		// Since in LLG Solver, all images are independent, we have one bool per image
		std::vector<bool> isConverged;
	};
}
#endif