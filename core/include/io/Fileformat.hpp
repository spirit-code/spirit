#pragma once
#ifndef SPIRIT_IO_FILEFORMAT_HPP
#define SPIRIT_IO_FILEFORMAT_HPP

#include <Spirit/IO.h>

#include <ostream>
#include <type_traits>

namespace IO
{

// A variety of supported file formats for vector fields
enum class VF_FileFormat
{
    // OOMF Vector Field file format
    OVF_BIN  = IO_Fileformat_OVF_bin,
    OVF_BIN4 = IO_Fileformat_OVF_bin4,
    OVF_BIN8 = IO_Fileformat_OVF_bin8,
    OVF_TEXT = IO_Fileformat_OVF_text,
    OVF_CSV  = IO_Fileformat_OVF_csv
};

inline std::string str( IO::VF_FileFormat format )
{
    if( format == IO::VF_FileFormat::OVF_BIN )
        return "binary OVF";
    else if( format == IO::VF_FileFormat::OVF_BIN4 )
        return "binary-4 OVF";
    else if( format == IO::VF_FileFormat::OVF_BIN8 )
        return "binary-8 OVF";
    else if( format == IO::VF_FileFormat::OVF_TEXT )
        return "text OVF";
    else if( format == IO::VF_FileFormat::OVF_CSV )
        return "CSV OVF";
    else
        return "unknown";
}

};

#endif
