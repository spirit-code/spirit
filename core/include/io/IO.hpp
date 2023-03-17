#pragma once
#ifndef SPIRIT_CORE_IO_IO_HPP
#define SPIRIT_CORE_IO_IO_HPP

#include <Spirit/Spirit_Defines.h>
#include <io/Fileformat.hpp>

#include <string>
#include <vector>

namespace IO
{

// Overwrites the file with the given string
void write_to_file( const std::string & str, const std::string & filename );

// Appends the string to a file
void append_to_file( const std::string & str, const std::string & filename );

/*
 * Writes the given string to a file, but may create and detach a thread, if
 * CORE_USE_THREADS is defined to do it asynchronously (i.e. fire & forget)
 */
void dump_to_file( const std::string & str, const std::string & filename );

} // namespace IO

#endif

// This is a single-include setup
#include "Configparser.hpp"
#include "Configwriter.hpp"
#include "Dataparser.hpp"
#include "Datawriter.hpp"
