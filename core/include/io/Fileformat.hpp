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
        SPIRIT_CSV_POS_SPIN        = IO_Fileformat_CSV_Pos,
        // Comma-separated values for orientation
        SPIRIT_CSV_SPIN            = IO_Fileformat_CSV,
        // Whitespace-separated values for position and orientation
        SPIRIT_WHITESPACE_POS_SPIN = IO_Fileformat_Regular_Pos,
        // Whitespace-separated values for orientation
        SPIRIT_WHITESPACE_SPIN     = IO_Fileformat_Regular,
        // OOMF Vector Field file format
        OVF_BIN8                   = IO_Fileformat_OVF_bin8,
        OVF_BIN4                   = IO_Fileformat_OVF_bin4,
        OVF_TEXT                   = IO_Fileformat_OVF_text,
        OVF_CSV                    = IO_Fileformat_OVF_csv,
        // General Spirit file
        GENERAL_TXT                = IO_Fileformat_GEN_text,
        GENERAL_CSV                = IO_Fileformat_GEN_csv,
        SPIRIT_GENERAL
    };
};

#endif
