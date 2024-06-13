#pragma once
#ifndef SPIRIT_CORE_UTILITY_CONFIGURATIONS_HPP
#define SPIRIT_CORE_UTILITY_CONFIGURATIONS_HPP

#include <Spirit/Spirit_Defines.h>
#include <data/Geometry.hpp>
#include <engine/Vectormath_Defines.hpp>

#include <functional>
#include <random>
#include <vector>

namespace Utility
{

namespace Configurations
{

using filterfunction               = std::function<bool( const Vector3 & )>;
const filterfunction defaultfilter = []( const Vector3 & pos ) -> bool { return true; };
void filter_to_mask( const vectorfield & positions, filterfunction filter, intfield & mask );

void Move( vectorfield & configuration, const Data::Geometry & geometry, int da, int db, int dc );

// Insert data in certain region
void Insert(
    vectorfield & spins, const Data::Geometry & geometry, const vectorfield & configuration, int shift = 0,
    filterfunction filter = defaultfilter );

// orients all spins with x>pos into the direction of the v
void Domain(
    vectorfield & spins, const Data::Geometry & geometry, Vector3 direction, filterfunction filter = defaultfilter );

// points all Spins in random directions
template<typename RandomFunc>
void Random_Sphere(
    vectorfield & spins, const Data::Geometry & geometry, RandomFunc & prng, filterfunction filter = defaultfilter );

template<typename RandomFunc>
void Random_Cube(
    vectorfield & field, const Data::Geometry & geometry, RandomFunc & prng, filterfunction filter = defaultfilter );

// Add temperature-scaled random noise to a system
template<typename RandomFunc>
void Add_Noise_Temperature_Sphere(
    vectorfield & spins, const Data::Geometry & geometry, scalar temperature, RandomFunc & prng,
    filterfunction filter = defaultfilter );

template<typename RandomFunc>
void Add_Noise_Temperature_Cube(
    vectorfield & field, const Data::Geometry & geometry, scalar temperature, RandomFunc & prng,
    filterfunction filter = defaultfilter );

// Creates a toroid
void Hopfion(
    vectorfield & spins, const Data::Geometry & geometry, Vector3 pos, scalar r, int order = 1,
    Vector3 normal = { 0, 0, 1 }, filterfunction filter = defaultfilter );

// Creates a Skyrmion
void Skyrmion(
    vectorfield & spins, const Data::Geometry & geometry, Vector3 pos, scalar r, scalar order, scalar phase,
    bool upDown, bool achiral, bool rl, bool experimental, filterfunction filter = defaultfilter );

// Creates a Skyrmion, following the circular domain wall ("swiss knife") profile
void DW_Skyrmion(
    vectorfield & spins, const Data::Geometry & geometry, Vector3 pos, scalar dw_radius, scalar dw_width, scalar order,
    scalar phase, bool upDown, bool achiral, bool rl, filterfunction filter = defaultfilter );

// Spin Spiral
void SpinSpiral(
    vectorfield & spins, const Data::Geometry & geometry, std::string direction_type, Vector3 q, Vector3 axis,
    scalar theta, filterfunction filter = defaultfilter );

// 2q Spin Spiral
void SpinSpiral(
    vectorfield & spins, const Data::Geometry & geometry, std::string direction_type, Vector3 q1, Vector3 q2,
    Vector3 axis, scalar theta, filterfunction filter = defaultfilter );

// Set atom types within a region of space
void Set_Atom_Types( Data::Geometry & geometry, int atom_type = 0, filterfunction filter = defaultfilter );

// Set spins to be pinned
void Set_Pinned(
    Data::Geometry & geometry, const vectorfield & spins, bool pinned, filterfunction filter = defaultfilter );

} // namespace Configurations

} // namespace Utility

#include <utility/Configurations.inl>

#endif
