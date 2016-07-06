#pragma once
#ifndef FORCE_GNEB_H
#define FORCE_GNEB_H

#include "Force.h"

namespace Engine
{
	class Force_GNEB : public Force
	{
	public:
		Force_GNEB(std::shared_ptr<Data::Spin_System_Chain> c);

		void Calculate(std::vector<std::vector<double>> & configurations, std::vector<std::vector<double>> & forces) override;

		bool IsConverged() override;

	private:
		bool isConverged;
	};

}
#endif