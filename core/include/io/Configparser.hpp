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

namespace IO
{

/*
 * Note that due to the modular structure of the input parsers, input may be given in one or in separate files.
 * Input may be given incomplete. In this case a log entry is created and default values are used.
 */

void Log_from_Config( const std::string & config_file_name, bool force_quiet = false );

std::unique_ptr<::State::system_t> Spin_System_from_Config( const std::string & config_file_name );

Data::Pinning Pinning_from_Config( const std::string & config_file_name, std::size_t n_cell_atoms );

Data::Geometry Geometry_from_Config( const std::string & config_file_name );

intfield Boundary_Conditions_from_Config( const std::string & config_file_name );

std::unique_ptr<Data::Parameters_Method_LLG> Parameters_Method_LLG_from_Config( const std::string & config_file_name );

std::unique_ptr<Data::Parameters_Method_MC> Parameters_Method_MC_from_Config( const std::string & config_file_name );

std::unique_ptr<Data::Parameters_Method_GNEB>
Parameters_Method_GNEB_from_Config( const std::string & config_file_name );

std::unique_ptr<Data::Parameters_Method_EMA> Parameters_Method_EMA_from_Config( const std::string & config_file_name );

std::unique_ptr<Data::Parameters_Method_MMF> Parameters_Method_MMF_from_Config( const std::string & config_file_name );

} // namespace IO

#endif
