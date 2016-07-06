#pragma once
#ifndef FORCE_LLG_H
#define FORCE_LLG_H

#include "Force.h"

namespace Engine
{
	class Force_MMF : public Force
	{
		void Calculate(std::vector<std::vector<double>> & configurations, std::vector<std::vector<double>> & forces) override {};
	};

}
#endif