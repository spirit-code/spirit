#pragma once
#ifndef UTILITY_CONFIGURATIONS_H
#define UTILITY_CONFIGURATIONS_H

#include "Core_Defines.h"
#include <data/Spin_System.hpp>

#include <vector>
#include <random>

namespace Utility
{
	namespace Configurations
	{
		// TODO: replace the Spin_System references with smart pointers??
		
		// orients all spins with x>pos into the direction of the v
		void DomainWall(Data::Spin_System &s, Vector3 pos, Vector3 v, const bool greater = true);

		// points all Spins parallel to the direction of v
		// calls DomainWall (s, -1E+20, v)
		void Homogeneous(Data::Spin_System &s, Vector3 v);
		// points all Spins in +z direction
		void PlusZ(Data::Spin_System &s);
		// points all Spins in -z direction
		void MinusZ(Data::Spin_System &s);

		// points all Spins in random directions
		void Random(Data::Spin_System &s, bool external = false);
		// points only spin no into a random direction created by prng
		void Random(Data::Spin_System &s, int no, std::mt19937 &prng);
		// Add temperature-scaled random noise to a system
		void Add_Noise_Temperature(Data::Spin_System & s, scalar temperature, int delta_seed=0);

		// Creates a toroid
		void Hopfion(Data::Spin_System & s, Vector3 pos, scalar r, int order=1);
		// points a sperical region of spins of radius r
		// into direction of vec at position pos
		void Skyrmion(Data::Spin_System & s, Vector3 pos, scalar r, scalar speed, scalar order, bool upDown, bool achiral, bool rl, bool experimental);
		// Spin Spiral
		void SpinSpiral(Data::Spin_System & s, std::string direction_type, Vector3 q, Vector3 axis, scalar theta);
	};//end namespace Configurations
}//end namespace Utility

#endif
