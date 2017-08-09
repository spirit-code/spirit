#pragma once
#ifndef UTILITY_CONFIGURATIONS_H
#define UTILITY_CONFIGURATIONS_H

#include "Spirit_Defines.h"
#include <data/Spin_System.hpp>

#include <vector>
#include <random>
#include <functional>

namespace Utility
{
	namespace Configurations
	{
		// Default filter function
		typedef std::function< bool(const Vector3&, const Vector3&) > filterfunction;
		filterfunction const defaultfilter = [](const Vector3& spin, const Vector3& pos)->bool { return true; };
		void filter_to_mask(const vectorfield & spins, const vectorfield & spin_pos, filterfunction filter, intfield & mask);

		// TODO: replace the Spin_System references with smart pointers??

		void Move(vectorfield& configuration, const Data::Geometry & geometry, int da, int db, int dc);

		// Insert data in certain region
		void Insert(Data::Spin_System &s, const vectorfield& configuration, int shift = 0, filterfunction filter = defaultfilter);

		// orients all spins with x>pos into the direction of the v
		void Domain(Data::Spin_System &s, Vector3 direction, filterfunction filter=defaultfilter);

		// points all Spins in random directions
		void Random(Data::Spin_System &s, filterfunction filter=defaultfilter, bool external = false);
		// Add temperature-scaled random noise to a system
		void Add_Noise_Temperature(Data::Spin_System & s, scalar temperature, int delta_seed=0, filterfunction filter=defaultfilter);

		// Creates a toroid
		void Hopfion(Data::Spin_System & s, Vector3 pos, scalar r, int order=1, filterfunction filter=defaultfilter);
		// points a sperical region of spins of radius r
		// into direction of vec at position pos
		void Skyrmion(Data::Spin_System & s, Vector3 pos, scalar r, scalar speed, scalar order, bool upDown, bool achiral, bool rl, bool experimental, filterfunction filter=defaultfilter);
		// Spin Spiral
		void SpinSpiral(Data::Spin_System & s, std::string direction_type, Vector3 q, Vector3 axis, scalar theta, filterfunction filter=defaultfilter);
		// 2q Spin Spiral
		void SpinSpiral(Data::Spin_System & s, std::string direction_type, Vector3 q1, Vector3 q2, Vector3 axis, scalar theta, filterfunction filter);
	};//end namespace Configurations
}//end namespace Utility

#endif
