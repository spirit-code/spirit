#pragma once

#include <data/Geometry.hpp>
#include <engine/Vectormath_Defines.hpp>
#include <engine/spin/Hamiltonian.hpp>

#include <memory>
#include <string>

namespace IO
{

template<typename Hamiltonian>
std::unique_ptr<Hamiltonian>
Hamiltonian_from_Config( const std::string & config_file_name, Data::Geometry geometry, intfield boundary_conditions );

} // namespace IO
