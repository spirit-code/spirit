#pragma once
#ifndef IO_OVFFILE_H
#define IO_OVFFILE_H

#include <io/IO.hpp>
#include <io/Fileformat.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Vectormath_Defines.hpp>
#include <utility/Logging.hpp>
#include <utility/Version.hpp>

#include <string>
#include <fstream>
#include <cctype>
    
#include <fmt/format.h>
#include <fmt/ostream.h>

namespace IO
{
    class OVF_File
    {
    private:
        VF_FileFormat format;
        std::string filename;
        std::string output_to_file;
        std::string empty_line;
        std::string datatype;
        std::string comment;
        int n_segments;
        uint32_t hex_4b_test = 0x4996B438;
        uint64_t hex_8b_test = 0x42DC12218377DE40;
       
        void Write_Top_Header( const int n_segments );
        void Write_Segment( const vectorfield& vf, const Data::Geometry& geometry );
        void Write_Segment_Data( const vectorfield& vf, const Data::Geometry& geometry );
        void Write_Data_bin( const vectorfield& vf );
        void Write_Data_txt( const vectorfield& vf );
        void Read_N_Segments();
    public:
        // Constructor
        OVF_File( std::string filename, VF_FileFormat format, const std::string comment );
        void write_image( const vectorfield& vf, const Data::Geometry& geometry );
        void write_eigenmodes( const std::vector<std::shared_ptr<vectorfield>>& modes, 
                               const Data::Geometry& geometry );
        void write_chain( const std::shared_ptr<Data::Spin_System_Chain>& chain );
    }; // end class OVF_File
} // end namespace io

#endif
