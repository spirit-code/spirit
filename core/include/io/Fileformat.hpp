#pragma once
#ifndef SPIRIT_CORE_IO_FILEFORMAT_HPP
#define SPIRIT_CORE_IO_FILEFORMAT_HPP

#include <Spirit/IO.h>
#include <utility/Enum.hpp>

#include <array>
#include <ostream>
#include <type_traits>

namespace IO
{

// The supported OOMF Vector Field (OVF) file formats
enum class VF_FileFormat
{
    // OVF
    OVF_BIN  = IO_Fileformat_OVF_bin,
    OVF_BIN4 = IO_Fileformat_OVF_bin4,
    OVF_BIN8 = IO_Fileformat_OVF_bin8,
    OVF_TEXT = IO_Fileformat_OVF_text,
    OVF_CSV  = IO_Fileformat_OVF_csv,
    // VTK
    VTK_HDF      = IO_Fileformat_VTK_hdf,
    VTK_XML_BIN  = IO_Fileformat_VTK_XML_bin,
    VTK_XML_TEXT = IO_Fileformat_VTK_XML_text,
};

inline constexpr auto enum_table( IO::VF_FileFormat )
{
    using E = Utility::Enum::TableElementType<IO::VF_FileFormat>;
    return std::array{
        // clang-format off
        E{ VF_FileFormat::OVF_BIN,  "ovf_bin",  "binary OVF",   "binary OVF"   },
        E{ VF_FileFormat::OVF_BIN4, "ovf_bin4", "binary-4 OVF", "binary-4 OVF" },
        E{ VF_FileFormat::OVF_BIN8, "ovf_bin8", "binary-8 OVF", "binary-8 OVF" },
        E{ VF_FileFormat::OVF_TEXT, "ovf_text", "text OVF",     "text OVF"     },
        E{ VF_FileFormat::OVF_CSV,  "ovf_csv",  "CSV OVF",      "CSV OVF"      },

        E{ VF_FileFormat::VTK_HDF,     "vtk_hdf",      "HDF5 (VTK)",       "HDF5 (VTK)" },
        E{ VF_FileFormat::VTK_XML_BIN, "vtk_xml_bin",  "XML binary (VTK)", "XML with binary data (VTK)" },
        E{ VF_FileFormat::VTK_XML_TEXT,"vtk_xml_text", "XML ascii (VTK)",  "XML with ascii data (VTK)" },
        // clang-format on
    };
}

} // namespace IO

#endif
