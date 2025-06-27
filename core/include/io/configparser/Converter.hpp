#pragma once
#ifndef SPIRIT_IO_CONFIGPARSER_CONVERTER
#define SPIRIT_IO_CONFIGPARSER_CONVERTER
#include <io/Configparser.hpp>

#include <toml++/toml.hpp>

#include <string>

namespace IO
{

namespace convert
{

auto Logging( const std::string & config_file_name ) -> toml::table;

auto Parameters_Method_EMA( const std::string & config_file_name ) -> toml::table;
auto Parameters_Method_GNEB( const std::string & config_file_name ) -> toml::table;
auto Parameters_Method_MMF( const std::string & config_file_name ) -> toml::table;
auto Parameters_Method_LLG( const std::string & config_file_name ) -> toml::table;
auto Parameters_Method_MC( const std::string & config_file_name ) -> toml::table;

auto Bravais_Vectors( const std::string & config_file_name ) -> toml::table;
auto Boundary_Conditions( const std::string & config_file_name ) -> toml::table;
auto Pinning( const std::string & config_file_name, std::size_t n_cell_atoms ) -> toml::table;
auto Geometry( const std::string & config_file_name ) -> toml::table;

namespace Interaction
{

auto Anisotropy( const std::string & config_file_name ) -> toml::table;
auto Biaxial_Anisotropy( const std::string & config_file_name ) -> toml::table;
auto DDI( const std::string & config_file_name ) -> toml::table;
auto Gaussian( const std::string & config_file_name ) -> toml::table;
auto Pair_Interactions_from_Pairs( const std::string & config_file_name ) -> toml::table;
auto Pair_Interactions_from_Shells( const std::string & config_file_name ) -> toml::table;
auto Quadruplets( const std::string & config_file_name ) -> toml::table;
auto Zeeman( const std::string & config_file_name ) -> toml::table;

} // namespace Interaction

} // namespace convert

} // namespace IO
#endif
