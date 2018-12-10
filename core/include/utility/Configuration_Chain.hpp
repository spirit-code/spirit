#pragma once
#ifndef UTILITY_CONFIGURATION_CHAIN_H
#define UTILITY_CONFIGURATION_CHAIN_H

#include "Spirit_Defines.h"
#include <data/Spin_System_Chain.hpp>

#include <vector>

namespace Utility
{
	namespace Configuration_Chain
	{
		// Add noise to the images of a transition (except the border images)
		void Add_Noise_Temperature(std::shared_ptr<Data::Spin_System_Chain> c, int idx_1, int idx_2, scalar temperature);

		// Homogeneous rotation of all spins from first to last configuration of the given configurations
		void Homogeneous_Rotation(std::shared_ptr<Data::Spin_System_Chain> c, int idx_1, int idx_2);

	};//end namespace Configurations
}//end namespace Utility

#endif
