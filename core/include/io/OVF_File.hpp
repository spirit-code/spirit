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
        // positions of the beggining of each segment in the input file 
        std::vector<std::ios::pos_type> segment_fpos;
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
      
        // Read OVF File version
        void Read_Version();
        // Read the "# Segments: " value and compare with the occurences of "# Begin: Segment "
        void Read_N_Segments();
        // Read binary OVF data
        void Read_Data_bin( vectorfield& vf );
        // Read text OVF data
        void Read_Data_txt( vectorfield& vf );
        // In case of binary data check the binary check values
        bool Read_Check_Binary_Values();
        // Read the segment's header
        void Read_Header();
        // Check if the geometry described in the last read header matches the given geometry
        void Check_Geometry( const Data::Geometry& geometry );
        // Read the segment's data
        void Read_Data( vectorfield& vf );
    public:
        // Constructor
        iFile_OVF( std::string filename, VF_FileFormat format );
        // Get the number of segments in the file
        int Get_N_Segments();
        // Read header and data from a given segment. Also check geometry
        void Read_Segment( vectorfield& vf, const Data::Geometry& geometry, 
                           const int idx_seg = 0 );
        // TODO: Changed that to Read_{String,Scalar,Int,Vec3}_from_Comments 
        void Read_Eigenvalue( scalar& eigenvalue, const int idx_seg = 0 );
    }; // end class iFile_OVF
} // end namespace io

#endif
