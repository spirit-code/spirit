#pragma once
#ifndef SPIRIT_CORE_IO_CONFIGPARSER_HPP
#define SPIRIT_CORE_IO_CONFIGPARSER_HPP

#include <data/Geometry.hpp>
#include <data/Parameters_Method_GNEB.hpp>
#include <data/Parameters_Method_LLG.hpp>
#include <data/Parameters_Method_MMF.hpp>
#include <data/Spin_System.hpp>
#include <data/Spin_System_Chain.hpp>
#include <engine/Hamiltonian.hpp>

namespace IO
{

/*
 * Note that due to the modular structure of the input parsers, input may be given in one or in separate files.
 * Input may be given incomplete. In this case a log entry is created and default values are used.
 */

void Log_from_Config( const std::string & config_file_name, bool force_quiet = false );

std::unique_ptr<Data::Spin_System> Spin_System_from_Config( const std::string & config_file_name );

Data::Pinning Pinning_from_Config( const std::string & config_file_name, std::size_t n_cell_atoms );

std::shared_ptr<Data::Geometry> Geometry_from_Config( const std::string & config_file_name );

intfield Boundary_Conditions_from_Config( const std::string & config_file_name );

std::unique_ptr<Data::Parameters_Method_LLG> Parameters_Method_LLG_from_Config( const std::string & config_file_name );

std::unique_ptr<Data::Parameters_Method_MC> Parameters_Method_MC_from_Config( const std::string & config_file_name );

std::unique_ptr<Data::Parameters_Method_GNEB>
Parameters_Method_GNEB_from_Config( const std::string & config_file_name );

std::unique_ptr<Data::Parameters_Method_EMA> Parameters_Method_EMA_from_Config( const std::string & config_file_name );

std::unique_ptr<Data::Parameters_Method_MMF> Parameters_Method_MMF_from_Config( const std::string & config_file_name );

std::unique_ptr<Engine::Hamiltonian> Hamiltonian_from_Config(
    const std::string & config_file_name, std::shared_ptr<Data::Geometry> geometry, intfield boundary_conditions );

std::unique_ptr<Engine::Hamiltonian> Hamiltonian_Heisenberg_from_Config(
    const std::string & config_file_name, std::shared_ptr<Data::Geometry> geometry, intfield boundary_conditions,
    const std::string & hamiltonian_type );

std::unique_ptr<Engine::Hamiltonian> Hamiltonian_Gaussian_from_Config(
    const std::string & config_file_name, std::shared_ptr<Data::Geometry> geometry, intfield boundary_conditions );

} // namespace IO

#endif
