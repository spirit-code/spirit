#pragma once
#ifndef IO_OVFFILE_H
#define IO_OVFFILE_H

#include <io/IO.hpp>
#include <io/Fileformat.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Vectormath_Defines.hpp>
#include <utility/Logging.hpp>
#include <utility/Version.hpp>
#include <io/Filter_File_Handle.hpp>

#include <string>
#include <fstream>
#include <cctype>
    
#include <fmt/format.h>
#include <fmt/ostream.h>

namespace IO
{
    class oFile_OVF
    {
    private:
        VF_FileFormat format;
        std::string filename;
        const uint32_t test_hex_4b = 0x4996B438;
        const uint64_t test_hex_8b = 0x42DC12218377DE40;
        Utility::Log_Sender sender;
        
        int n_segments;
        std::string output_to_file;
        std::string empty_line;
        std::string datatype_out;
        std::string comment;
        
        void Write_Top_Header( const int n_segments = 1 );
        void Write_Segment( const vectorfield& vf, const Data::Geometry& geometry,
                            const std::string& eigenvalue = "" );
        void Write_Data_bin( const vectorfield& vf );
        void Write_Data_txt( const vectorfield& vf );
    public:
        // Constructor
        oFile_OVF( std::string filename, VF_FileFormat format, const std::string comment = "" );
        void write_image( const vectorfield& vf, const Data::Geometry& geometry );
        void write_eigenmodes( const std::vector<scalar>& eigenvalues,
                               const std::vector<std::shared_ptr<vectorfield>>& modes, 
                               const Data::Geometry& geometry );
        void write_chain( const std::shared_ptr<Data::Spin_System_Chain>& chain );
    }; // end class oFile_OVF

    class iFile_OVF
    {
    private:
        VF_FileFormat format;
        std::string filename;
        const uint32_t test_hex_4b = 0x4996B438;
        const uint64_t test_hex_8b = 0x42DC12218377DE40;
        Utility::Log_Sender sender;
       
        int n_segments;
        Filter_File_Handle myfile;
        std::string version;
        std::string title;
        std::string meshunit;
        std::string meshtype;
        std::string valueunits;
        std::string datatype_in;
        int binary_length;
        Vector3 max;
        Vector3 min;
        int valuedim;
        // irregular mesh
        int pointcount;
        // rectangular mesh
        Vector3 base;
        Vector3 stepsize;
        std::array<int,3> nodes;
       
        void Read_Version();
        void Read_N_Segments();
        void Read_Header();
        void Read_Check_Geometry( const Data::Geometry& geometry);
        void Read_Eigenvalue( scalar& eigenvalue);
        void Read_Data( vectorfield& vf );
        void Read_Data_bin( vectorfield& vf );
        void Read_Data_txt( vectorfield& vf );
        bool Read_Check_Binary_Values();
    public:
        // Constructor
        iFile_OVF( std::string filename, VF_FileFormat format );
        void read_image( vectorfield& vf, Data::Geometry& geometry );
        void read_eigenmodes( std::vector<scalar>& eigenmodes,
                              std::vector<std::shared_ptr<vectorfield>>& modes, 
                              Data::Geometry& geometry );
    }; // end class iFile_OVF
} // end namespace io

#endif
