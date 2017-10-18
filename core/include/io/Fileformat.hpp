#pragma once
#ifndef IO_FILEFORMAT_H
#define IO_FILEFORMAT_H

#include <Spirit/IO.h>

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
        OVF                 = IO_Fileformat_OVF,
        // General Spirit file
        SPIRIT_GENERAL
    };
};

#endif