#pragma once
#ifndef UTILITY_CONFIGURATIONS_H
#define UTILITY_CONFIGURATIONS_H

#include "Spin_System.h"
#include <vector>
#include <random>

namespace Utility
{
	namespace Configurations
	{
		// TODO: replace the Spin_System references with smart pointers??
		
		// orients all spins with x>pos into the direction of the v
		void DomainWall(Data::Spin_System &s, const double pos[3], double v[3], const bool greater = true);

		// points all Spins parallel to the direction of v
		// calls DomainWall (s, -1E+20, v)
		void Homogeneous(Data::Spin_System &s, double v[3]);
		// points all Spins in +z direction
		void PlusZ(Data::Spin_System &s);
		// points all Spins in -z direction
		void MinusZ(Data::Spin_System &s);

		// points all Spins in random directions
		void Random(Data::Spin_System &s, bool external = false);
		// points only spin no into a random direction created by prng
		void Random(Data::Spin_System &s, int no, std::mt19937 &prng);
		// Add temperature-scaled random noise to a system
		void Add_Noise_Temperature(Data::Spin_System & s, double temperature, int delta_seed=0);

		// points a sperical region of spins of radius r
		// into direction of vec at position pos
		void Skyrmion(Data::Spin_System & s, std::vector<double> pos, double r, double speed, double order, bool upDown, bool achiral, bool rl, bool experimental);
		// Spin Spiral
		void SpinSpiral(Data::Spin_System & s, std::string direction_type, double q[3], double axis[3], double theta);
	};//end namespace Configurations
}//end namespace Utility

#endif
