#pragma once
#ifndef IO_FILEFORMAT_H
#define IO_FILEFORMAT_H

#include <Spirit/IO.h>

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
};

#endif
