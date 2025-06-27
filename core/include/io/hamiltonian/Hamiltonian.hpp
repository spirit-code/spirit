#pragma once

#include <data/Geometry.hpp>
#include <engine/Vectormath_Defines.hpp>
#include <engine/spin/Hamiltonian.hpp>

#include <toml++/toml.hpp>

#include <memory>

namespace IO
{

template<typename Hamiltonian>
std::unique_ptr<Hamiltonian> Hamiltonian_from_TOML( const toml::table & tbl, Data::Geometry geometry );

} // namespace IO
