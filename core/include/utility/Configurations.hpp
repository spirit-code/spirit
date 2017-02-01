#pragma once
#ifndef UTILITY_CONFIGURATIONS_H
#define UTILITY_CONFIGURATIONS_H

#include "Core_Defines.h"
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

		// TODO: replace the Spin_System references with smart pointers??
		
		// orients all spins with x>pos into the direction of the v
		void Domain(Data::Spin_System &s, Vector3 direction, filterfunction filter=defaultfilter);

		// points all Spins in random directions
		void Random(Data::Spin_System &s, filterfunction filter=defaultfilter, bool external = false);
		// points only spin no into a random direction created by prng
		void Random(Data::Spin_System &s, int no, std::mt19937 &prng);
		// Add temperature-scaled random noise to a system
		void Add_Noise_Temperature(Data::Spin_System & s, scalar temperature, int delta_seed=0, filterfunction filter=defaultfilter);

		// Creates a toroid
		void Hopfion(Data::Spin_System & s, Vector3 pos, scalar r, int order=1, filterfunction filter=defaultfilter);
		// points a sperical region of spins of radius r
		// into direction of vec at position pos
		void Skyrmion(Data::Spin_System & s, Vector3 pos, scalar r, scalar speed, scalar order, bool upDown, bool achiral, bool rl, bool experimental, filterfunction filter=defaultfilter);
		// Spin Spiral
		void SpinSpiral(Data::Spin_System & s, std::string direction_type, Vector3 q, Vector3 axis, scalar theta, filterfunction filter=defaultfilter);
	};//end namespace Configurations
}//end namespace Utility

#endif
