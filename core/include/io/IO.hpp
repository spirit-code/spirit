#pragma once
#ifndef IO_H
#define IO_H

#include <string>
#include <vector>

#include <Spirit/IO.h>
#include "Spirit_Defines.h"

namespace IO
{
    // A variety of supported file formats for vector fields
    enum class VF_FileFormat
    {
        // Comma-separated values for position and orientation
        CSV_POS_SPIN        = IO_Fileformat_CSV_Pos,
        // Comma-separated values for orientation
        CSV_SPIN            = IO_Fileformat_CSV,
        // Whitespace-separated values for position and orientation
        WHITESPACE_POS_SPIN = IO_Fileformat_Regular_Pos,
        // Whitespace-separated values for orientation
        WHITESPACE_SPIN     = IO_Fileformat_Regular,
        // OOMF Vector Field file format
        OVF                 = IO_Fileformat_OVF
    };
    
    // ------ Saving Helpers --------------------------------------------
	// Creates a new thread with String_to_File, which is immediately detached
	void Dump_to_File(const std::string text, const std::string name);
	// Takes a vector of strings of size "no" and dumps those into a file asynchronously
	void Dump_to_File(const std::vector<std::string> text, const std::string name, const int no);

	// Dumps the contents of the strings in text vector into file "name"
	void Strings_to_File(const std::vector<std::string> text, const std::string name, const int no);
	// Dumps the contents of the string 'text' into a file
	void String_to_File(const std::string text, const std::string name);
	// Appends the contents of the string 'text' onto a file
	void Append_String_to_File(const std::string text, const std::string name);
    // ------------------------------------------------------------------

};// end namespace IO

#endif

// This is a single-include setup
#include <Configparser.hpp>
#include <Configwriter.hpp>
#include <Dataparser.hpp>
#include <io/Datawriter.hpp>