#pragma once
#ifndef SPIRIT_CORE_IO_CONFIGPARSER_HPP
#define SPIRIT_CORE_IO_CONFIGPARSER_HPP

#include <data/Geometry.hpp>
#include <data/Parameters_Method_GNEB.hpp>
#include <data/Parameters_Method_LLG.hpp>
#include <data/Parameters_Method_MMF.hpp>
#include <data/State.hpp>
#include <engine/spin/Hamiltonian.hpp>
#include <io/hamiltonian/Hamiltonian.hpp>

#include <toml++/toml.hpp>

namespace IO
{

namespace detail
{

struct BravaisConfig
{
    std::vector<Vector3> vectors          = std::vector{ Vector3{ 1, 0, 0 }, Vector3{ 0, 1, 0 }, Vector3{ 0, 0, 1 } };
    Data::BravaisLatticeType lattice_type = Data::BravaisLatticeType::SC;
    std::string lattice_type_str          = "sc";
};

} // namespace detail
/*
 * Note that due to the modular structure of the input parsers, input may be given in one or in separate files.
 * Input may be given incomplete. In this case a log entry is created and default values are used.
 */

void Log_from_Config( const std::string & config_file_name, bool force_quiet = false );
void Log_from_TOML( const toml::table & tbl, bool force_quiet = false );

std::unique_ptr<::State::system_t> Spin_System_from_Config( const std::string & config_file_name );

auto Geometry_from_Config( const std::string & config_file_name ) -> Data::Geometry;
auto Boundary_Conditions_from_Config( const std::string & config_file_name ) -> intfield;

auto Parameters_Method_LLG_from_Config( const std::string & ) -> std::unique_ptr<Data::Parameters_Method_LLG>;
auto Parameters_Method_LLG_from_TOML( const toml::table & ) -> std::unique_ptr<Data::Parameters_Method_LLG>;

auto Parameters_Method_MC_from_Config( const std::string & ) -> std::unique_ptr<Data::Parameters_Method_MC>;
auto Parameters_Method_MC_from_TOML( const toml::table & ) -> std::unique_ptr<Data::Parameters_Method_MC>;

auto Parameters_Method_GNEB_from_Config( const std::string & ) -> std::unique_ptr<Data::Parameters_Method_GNEB>;
auto Parameters_Method_GNEB_from_TOML( const toml::table & ) -> std::unique_ptr<Data::Parameters_Method_GNEB>;

auto Parameters_Method_EMA_from_Config( const std::string & ) -> std::unique_ptr<Data::Parameters_Method_EMA>;
auto Parameters_Method_EMA_from_TOML( const toml::table & ) -> std::unique_ptr<Data::Parameters_Method_EMA>;

auto Parameters_Method_MMF_from_Config( const std::string & ) -> std::unique_ptr<Data::Parameters_Method_MMF>;
auto Parameters_Method_MMF_from_TOML( const toml::table & ) -> std::unique_ptr<Data::Parameters_Method_MMF>;

} // namespace IO

#include <io/Configparser.inl>

#endif
