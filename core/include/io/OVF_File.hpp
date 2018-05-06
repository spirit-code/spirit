#pragma once
#ifndef IO_OVFFILE_H
#define IO_OVFFILE_H

#include <io/IO.hpp>
#include <io/Fileformat.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Vectormath_Defines.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>
#include <utility/Version.hpp>
#include <io/Filter_File_Handle.hpp>

#include <string>
#include <fstream>
#include <cctype>
    
#include <fmt/format.h>
#include <fmt/ostream.h>

namespace IO
{
    class File_OVF
    {
    private:
        bool isOVF;
        VF_FileFormat format;
        std::string filename;
        const uint32_t test_hex_4b = 0x4996B438;
        const uint64_t test_hex_8b = 0x42DC12218377DE40;
        const std::string comment_tag = "##";
        Utility::Log_Sender sender;
        
        int n_segments;
        std::string n_segments_as_str;
        std::ios::pos_type n_segments_pos; 
        const int n_segments_str_digits = 6;  // can store 1M modes
        bool file_exists; 
        // Positions of the beggining of each segment in the input file 
        std::vector<std::ios::pos_type> segment_fpos;

        // Output attributes
        const std::string empty_line = "#\n";
        std::string output_to_file;
        std::string datatype_out;

        // Input attributes 
        std::unique_ptr<Filter_File_Handle> ifile;
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
        // Irregular mesh
        int pointcount;
        // Rectangular mesh
        std::array<Vector3,3> base;
        Vector3 stepsize;
        std::array<int,3> nodes;

        // Check OVF version
        void check_version();
        // Read segment's header
        void read_header();
        // Check segment's geometry
        void check_geometry( const Data::Geometry& geometry );
        // Read segment's data
        void read_data( vectorfield& vf, Data::Geometry& geometry );
        // In case of binary data check the binary check values
        bool check_binary_values();
        // Read binary OVF data
        void read_data_bin( vectorfield& vf, Data::Geometry& geometry );
        // Read text OVF data. The delimiter, if any, will be discarded in the reading
        void read_data_txt( vectorfield& vf, Data::Geometry& geometry, 
                            const std::string& delimiter = "" );
        // Write OVF file header
        void write_top_header();
        // Write segment data binary
        void write_data_bin( const vectorfield& vf );
        // Write segment data text
        void write_data_txt( const vectorfield& vf, const std::string& delimiter = "" ); 
        // Increment segment count
        void increment_n_segments();
        // Read the number of segments in the file by reading the top header
        void read_n_segments_from_top_header();
        // Count the number of segments in the file. It also saves their file positions
        int count_and_locate_segments();

    public:
        // constructor
        File_OVF( std::string filename, VF_FileFormat format = VF_FileFormat::OVF_TEXT  );
        // Check if the file is in OVF format
        bool is_OVF();
        // Get the number of segments in the file
        int get_n_segments();
        // Read header and data from a given segment. Also check geometry
        void read_segment( vectorfield& vf, Data::Geometry& geometry, 
                           const int idx_seg = 0 );
        // Write segment to file (if the file exists overwrite it)
        void write_segment( const vectorfield& vf, const Data::Geometry& geometry,
                            const std::string comment = "", const bool append = false );

        void write_eigenmodes( const std::vector<scalar>& eigenvalues,
                               const std::vector<std::shared_ptr<vectorfield>>& modes,
                               const Data::Geometry& geometry );
 
        // Read a variable from the comment section from the header of segment idx_seg
        template <typename T> void Read_Variable_from_Comment( T& var, const std::string name,
                                                               const int idx_seg = 0 )
        {
            try
            {
                // NOTE: seg_idx.max = segment_fpos.size - 2
                if ( idx_seg >= ( this->segment_fpos.size() - 1 ) )
                    spirit_throw( Utility::Exception_Classifier::Input_parse_failed, 
                                  Utility::Log_Level::Error,
                                  "OVF error while choosing segment to read - "
                                  "index out of bounds" );

                this->ifile->SetLimits( this->segment_fpos[idx_seg], 
                                        this->segment_fpos[idx_seg+1] );

                this->ifile->Read_Single( var, name );

                Log( Utility::Log_Level::Debug, this->sender, fmt::format( "{}{}", name, var ) );
            }
            catch (...)
            {
                spirit_handle_exception_core(fmt::format( "Failed to read variable \"{}\" "
                                                          "from comment", name ));
            }
        }
    
        // Read a Vector3 from the comment section from the header of segment idx_seg
        template <typename T> void Read_String_from_Comment( T& var, std::string name,
                                                              const int idx_seg = 0 )
        {
            try
            {
                // NOTE: seg_idx.max = segment_fpos.size - 2
                if ( idx_seg >= ( this->segment_fpos.size() - 1 ) )
                    spirit_throw( Utility::Exception_Classifier::Input_parse_failed, 
                                  Utility::Log_Level::Error,
                                  "OVF error while choosing segment to read - "
                                  "index out of bounds" );

                this->ifile->SetLimits( this->segment_fpos[idx_seg], 
                                        this->segment_fpos[idx_seg+1] );

                this->ifile->Read_String( var, name, false );

                Log( Utility::Log_Level::Debug, this->sender, fmt::format( "{}{}", name, var ) );
            }
            catch (...)
            {
                spirit_handle_exception_core(fmt::format( "Failed to read string \"{}\" "
                                                          "from comment", name ));
            }
        }
    };

} // end namespace io

#endif
