#pragma once
#ifndef SPIRIT_CORE_UTILITY_CONFIGURATIONS_HPP
#define SPIRIT_CORE_UTILITY_CONFIGURATIONS_HPP

#include "Spirit_Defines.h"
#include <data/Spin_System.hpp>

#include <functional>
#include <random>
#include <vector>

namespace Utility
{
namespace Configurations
{

// Default filter function
using filterfunction               = std::function<bool( const Vector3 &, const Vector3 & )>;
const filterfunction defaultfilter = []( const Vector3 & spin, const Vector3 & pos ) -> bool { return true; };
void filter_to_mask( const vectorfield & spins, const vectorfield & positions, filterfunction filter, intfield & mask );

// TODO: replace the Spin_System references with smart pointers??

void Move( vectorfield & configuration, const Data::Geometry & geometry, int da, int db, int dc );

// Insert data in certain region
void Insert(
    Data::Spin_System & s, const vectorfield & configuration, int shift = 0, filterfunction filter = defaultfilter );

// orients all spins with x>pos into the direction of the v
void Domain( Data::Spin_System & s, Vector3 direction, filterfunction filter = defaultfilter );

// points all Spins in random directions
void Random( Data::Spin_System & s, filterfunction filter = defaultfilter, bool external = false );

// Add temperature-scaled random noise to a system
void Add_Noise_Temperature(
    Data::Spin_System & s, scalar temperature, int delta_seed = 0, filterfunction filter = defaultfilter );

// Creates a toroid
void Hopfion( Data::Spin_System & s, Vector3 pos, scalar r, int order = 1, Vector3 normal = {0,0,1}, filterfunction filter = defaultfilter );

// Creates a Skyrmion
void Skyrmion(
    Data::Spin_System & s, Vector3 pos, scalar r, scalar order, scalar phase, bool upDown, bool achiral, bool rl,
    bool experimental, filterfunction filter = defaultfilter );

// Creates a Skyrmion, following the circular domain wall ("swiss knife") profile
void DW_Skyrmion(
    Data::Spin_System & s, Vector3 pos, scalar dw_radius, scalar dw_width, scalar order, scalar phase, bool upDown,
    bool achiral, bool rl, filterfunction filter = defaultfilter );

// Spin Spiral
void SpinSpiral(
    Data::Spin_System & s, std::string direction_type, Vector3 q, Vector3 axis, scalar theta,
    filterfunction filter = defaultfilter );

// 2q Spin Spiral
void SpinSpiral(
    Data::Spin_System & s, std::string direction_type, Vector3 q1, Vector3 q2, Vector3 axis, scalar theta,
    filterfunction filter = defaultfilter );

// Set atom types within a region of space
void Set_Atom_Types( Data::Spin_System & s, int atom_type = 0, filterfunction filter = defaultfilter );

// Set spins to be pinned
void Set_Pinned( Data::Spin_System & s, bool pinned, filterfunction filter = defaultfilter );

} //  namespace Configurations
} //  namespace Utility

#endif