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

}

} // namespace IO
#endif
