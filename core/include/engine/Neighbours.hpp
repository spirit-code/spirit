#pragma once
#ifndef SPIRIT_CORE_ENGINE_NEIGHBOURS_HPP
#define SPIRIT_CORE_ENGINE_NEIGHBOURS_HPP

#include <Spirit/Spirit_Defines.h>
#include <data/Geometry.hpp>
#include <engine/Vectormath_Defines.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace Engine
{
namespace Neighbours
{

std::vector<scalar> Get_Shell_Radii( const Data::Geometry & geometry, std::size_t n_shells );

pairfield Get_Pairs_in_Radius( const Data::Geometry & geometry, scalar radius );

void Get_Neighbours_in_Shells(
    const Data::Geometry & geometry, std::size_t n_shells, pairfield & neighbours, intfield & shells,
    bool use_redundant_neighbours );

Vector3 DMI_Normal_from_Pair( const Data::Geometry & geometry, const Pair & pair, std::int8_t chirality = 1 );

void DDI_from_Pair( const Data::Geometry & geometry, const Pair & pair, scalar & magnitude, Vector3 & normal );

} // namespace Neighbours
} // namespace Engine

#endif