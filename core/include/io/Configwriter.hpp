#pragma once
#ifndef SPIRIT_CORE_IO_CONFIGWRITER_HPP
#define SPIRIT_CORE_IO_CONFIGWRITER_HPP

#include <data/Geometry.hpp>
#include <data/Parameters_Method_GNEB.hpp>
#include <data/Parameters_Method_LLG.hpp>
#include <data/Parameters_Method_MMF.hpp>
#include <data/Spin_System.hpp>
#include <data/Spin_System_Chain.hpp>
#include <engine/spin/Hamiltonian.hpp>

#include <toml++/toml.hpp>

namespace IO
{

auto Logging_to_TOML() -> toml::table;
auto Geometry_to_TOML( const Data::Geometry & ) -> toml::table;
auto Hamiltonian_to_TOML( const Engine::Spin::Hamiltonian & ) -> toml::table;

auto Parameters_Method_LLG_to_TOML( const Data::Parameters_Method_LLG & ) -> toml::table;
auto Parameters_Method_MC_to_TOML( const Data::Parameters_Method_MC & ) -> toml::table;
auto Parameters_Method_GNEB_to_TOML( const Data::Parameters_Method_GNEB & ) -> toml::table;
auto Parameters_Method_EMA_to_TOML( const Data::Parameters_Method_EMA & ) -> toml::table;
auto Parameters_Method_MMF_to_TOML( const Data::Parameters_Method_MMF & ) -> toml::table;

inline auto as_inline( toml::table && tbl ) -> toml::table
{
    tbl.is_inline( true );
    return tbl;
}

} // namespace IO

#endif
