#pragma once
#ifndef LIBOVF_DETAIL_WRITE_H
#define LIBOVF_DETAIL_WRITE_H

#include "ovf.h"
#include <detail/parse.hpp>

#include <fmt/format.h>

#include <string>
#include <vector>
#include <fstream>
#include <iostream>

namespace ovf
{
namespace detail
{
namespace write
{
    // Comment tag in OVF file header
    static const std::string comment_tag = "##";

    // ?
    static const std::string empty_line = "#\n";

    // The number of zero-padding for segment count.
    // This is needed so that, when appending, the file does not need to be overwritten
    static const int n_segments_str_digits = 6; // can store 1M modes

    class file_handle
    {
    private:
        std::ofstream myfile;
    public:
        file_handle(const std::string & filename, bool append);
        ~file_handle();
        void write(const std::vector<std::string> & text);
        void append(const std::vector<std::string> & text);
    };

    inline file_handle::file_handle( const std::string & filename, bool append )
    {
        if( append )
            myfile.open(filename, std::ofstream::out | std::ofstream::app);
        else
            myfile.open(filename);
    }


    inline file_handle::~file_handle()
    {
        myfile.close();
    }


    inline void file_handle::write(const std::vector<std::string> & text)
    {
        if( myfile.is_open() )
        {
            // Log(Log_Level::Debug, Log_Sender::All, "Started writing " + name);
            for (int i = 0; i < text.size(); ++i)
                myfile << text[i];

            myfile.close();
            // Log(Log_Level::Debug, Log_Sender::All, "Finished writing " + name);
        }
        else
        {
            // Log(Log_Level::Error, Log_Sender::All, "Could not open " + name + " to write to file");
        }
    }


    inline std::string top_header_string()
    {
        std::string ret = "# OOMMF OVF 2.0\n";
        ret += empty_line;

        // create padding string
        std::string padding( n_segments_str_digits, '0' );
        // write padding plus n_segments
        ret += fmt::format( "# Segment count: {}\n", padding );

        return ret;
    }


    inline int increment_n_segments(ovf_file *file)
    try
    {
        parse::file_header(*file);

        // Update n_segments
        file->n_segments++;

        // Convert updated n_segment into padded string
        std::string new_n_str = std::to_string( file->n_segments );
        std::string::size_type new_n_len = new_n_str.length();
        std::string::size_type padding_len = n_segments_str_digits - new_n_len;
        std::string padding( padding_len, '0' );

        // Replace n_segments value in the stream
        std::fstream filestream( file->file_name );
        filestream.seekg( file->_state->n_segments_pos );
        filestream << ( padding + new_n_str );
        filestream.close();

        return OVF_OK;
    }
    catch( ... )
    {
        file->_state->message_latest = fmt::format("increment_n_segments failed for file \"{}\".", file->file_name);
        return OVF_ERROR;
    }


    template <typename T>
    void append_data_bin_to_string( std::string & output_to_file, const T * vf, int n_cols, int n_rows, int format)
    try
    {
        if( format == OVF_FORMAT_BIN8 )
        {
            auto out_check = std::vector<uint8_t>(8);
            endian::to_little_64(check::val_8b, out_check.data());

            output_to_file +=
                std::string( reinterpret_cast<const char *>(out_check.data()), sizeof(double) );

            std::vector<uint8_t> out(n_cols*8);
            for (unsigned int i=0; i<n_rows; ++i)
            {
                for (int j=0; j<n_cols; ++j)
                {
                    double val = static_cast<double>(vf[n_cols*i + j]);
                    uint64_t in = *reinterpret_cast<uint64_t*>(&val);

                    endian::to_little_64(in, &out[j*8]);
                }
                output_to_file +=
                    std::string( reinterpret_cast<char *>(out.data()), n_cols*sizeof(double) );
            }
        }
        else if( format == OVF_FORMAT_BIN4 )
        {
            auto out_check = std::vector<uint8_t>(4);
            endian::to_little_32(check::val_4b, out_check.data());

            output_to_file +=
                std::string( reinterpret_cast<const char *>(out_check.data()), sizeof(float) );

            std::vector<uint8_t> out(n_cols*4);
            for (unsigned int i=0; i<n_rows; ++i)
            {
                for (int j=0; j<n_cols; ++j)
                {
                    float val = static_cast<float>(vf[n_cols*i + j]);
                    uint32_t in = *reinterpret_cast<uint32_t*>(&val);

                    endian::to_little_32(in, &out[j*4]);
                }
                output_to_file +=
                    std::string( reinterpret_cast<char *>(out.data()), n_cols*sizeof(float) );
            }
        }

        output_to_file += "\n";
    }
    catch( ... )
    {

    }


    template <typename T>
    void append_data_txt_to_string( std::string & output_to_file, const T * vf, int n_cols, int n_rows,
        const std::string& delimiter = "" )
    try
    {
        for (int row = 0; row < n_rows; ++row)
        {
            for (int col = 0; col < n_cols; ++col)
                output_to_file += fmt::format( "{:22.12f}{}", vf[n_cols*row + col], delimiter );
            output_to_file += "\n";
        }
    }
    catch( ... )
    {

    }


    template <typename T>
    int segment( ovf_file *file, const ovf_segment * segment, const T * vf,
                const bool append = false, int format = OVF_FORMAT_BIN8 )
    try
    {
        if( file->_state->file_contents.size() > 0 && append )
            file->_state->file_contents.push_back("");
        else
            file->_state->file_contents = {""};

        int index = file->_state->file_contents.size()-1;
        std::string & output_to_file = file->_state->file_contents[index];
        output_to_file.reserve( int( 0x08000000 ) );  // reserve 128[MByte]

        output_to_file += fmt::format( empty_line );
        output_to_file += fmt::format( "# Begin: Segment\n" );
        output_to_file += fmt::format( "# Begin: Header\n" );
        output_to_file += fmt::format( empty_line );

        output_to_file += fmt::format( "# Title: {}\n", segment->title );
        output_to_file += fmt::format( empty_line );

        output_to_file += fmt::format( "# Desc: {}\n", segment->comment );
        output_to_file += fmt::format( empty_line );

        // The value dimension is always 3 since we are writting Vector3-data
        output_to_file += fmt::format( "# valuedim: {}   ## field dimensionality\n", segment->valuedim );

        //
        if( std::string(segment->valueunits) == "" )
        {
            output_to_file += "# valueunits: ";
            for( int i=0; i<segment->valuedim; ++i )
                output_to_file += " unspecified";
            output_to_file += "\n";
        }
        else
            output_to_file += fmt::format( "# valueunits: {}\n", segment->valueunits );

        //
        if( std::string(segment->valuelabels) == "" )
        {
            output_to_file += "# valuelabels: ";
            for( int i=0; i<segment->valuedim; ++i )
                output_to_file += " unspecified";
            output_to_file += "\n";
        }
        else
            output_to_file += fmt::format( "# valuelabels: {}\n", segment->valuelabels );

        // spatial unit
        output_to_file += fmt::format( empty_line );
        output_to_file += fmt::format( "## Fundamental mesh measurement unit. Treated as a label:\n" );
        if( std::string(segment->meshunit) == "" )
            output_to_file += "# meshunit: unspecified\n";
        else
            output_to_file += fmt::format( "# meshunit: {}\n", segment->meshunit );

        // extent
        output_to_file += fmt::format( empty_line );
        output_to_file += fmt::format( "# xmin: {}\n", segment->bounds_min[0] );
        output_to_file += fmt::format( "# ymin: {}\n", segment->bounds_min[1] );
        output_to_file += fmt::format( "# zmin: {}\n", segment->bounds_min[2] );
        output_to_file += fmt::format( "# xmax: {}\n", segment->bounds_max[0] );
        output_to_file += fmt::format( "# ymax: {}\n", segment->bounds_max[1] );
        output_to_file += fmt::format( "# zmax: {}\n", segment->bounds_max[2] );
        output_to_file += fmt::format( empty_line );

        // Type of mesh and further keywords depending on it
        std::string meshtype = segment->meshtype;
        if( meshtype == "" )
            meshtype = "rectangular";
        output_to_file += fmt::format( "# meshtype: {}\n", meshtype );

        int n_rows = 0;
        if( meshtype == "rectangular" )
        {
            // Latice origin in space
            output_to_file += fmt::format( "# xbase: {}\n", segment->origin[0] );
            output_to_file += fmt::format( "# ybase: {}\n", segment->origin[1] );
            output_to_file += fmt::format( "# zbase: {}\n", segment->origin[2] );

            // Mesh spacing
            output_to_file += fmt::format( "# xstepsize: {}\n", segment->step_size[0] );
            output_to_file += fmt::format( "# ystepsize: {}\n", segment->step_size[1] );
            output_to_file += fmt::format( "# zstepsize: {}\n", segment->step_size[2] );

            // Number of nodes along each direction
            output_to_file += fmt::format( "# xnodes: {}\n", segment->n_cells[0] );
            output_to_file += fmt::format( "# ynodes: {}\n", segment->n_cells[1] );
            output_to_file += fmt::format( "# znodes: {}\n", segment->n_cells[2] );

            n_rows = segment->n_cells[0]*segment->n_cells[1]*segment->n_cells[2];
        }
        else if( std::string(segment->meshtype) == "irregular" )
        {
            output_to_file += fmt::format( "# pointcount: {}\n", segment->pointcount );
            n_rows = segment->pointcount;
        }
        else
        {
            file->_state->message_latest = fmt::format(
                "write_segment not writing out any data to file \"{}\", because meshtype is invalid: \"{}\". "
                "You may want to check the segment you passed in.",
                file->file_name, segment->meshtype);
            return OVF_ERROR;
        }

        int n_cols = segment->valuedim;

        // Check that we actually read in any data
        if( n_cols*n_rows <= 0 )
        {
            file->_state->message_latest = fmt::format(
                "write_segment not writing out any data, because n_cols*n_rows={}*{}<=0 for file \"{}\". "
                "You may want to check the segment you passed in.",
                n_cols, n_rows, file->file_name);
            return OVF_ERROR;
        }

        output_to_file += fmt::format( empty_line );
        output_to_file += fmt::format( "# End: Header\n" );
        output_to_file += fmt::format( empty_line );

        if( sizeof(T) == sizeof(float) && format == OVF_FORMAT_BIN )
            format = OVF_FORMAT_BIN4;
        else if( sizeof(T) == sizeof(double) && format == OVF_FORMAT_BIN )
            format = OVF_FORMAT_BIN8;

        std::string datatype_out = "";
        if ( format == OVF_FORMAT_BIN8 )
            datatype_out = "Binary 8";
        else if ( format == OVF_FORMAT_BIN4 )
            datatype_out = "Binary 4";
        else if( format == OVF_FORMAT_TEXT )
            datatype_out = "Text";
        else if( format == OVF_FORMAT_CSV )
            datatype_out = "CSV";

        // Data
        output_to_file += fmt::format( "# Begin: Data {}\n", datatype_out );

        if ( format == OVF_FORMAT_BIN || format == OVF_FORMAT_BIN8 || format == OVF_FORMAT_BIN4 )
            append_data_bin_to_string( output_to_file, vf, n_cols, n_rows, format );
        else if ( format == OVF_FORMAT_TEXT )
            append_data_txt_to_string( output_to_file, vf, n_cols, n_rows );
        else if ( format == OVF_FORMAT_CSV )
            append_data_txt_to_string( output_to_file, vf, n_cols, n_rows, "," );
        else
        {
            file->_state->message_latest = fmt::format(
                "write_segment not writing out any data, because format \"{}\" is invalid. "
                "You may want to check what you passed in.", format);
            return OVF_ERROR;
        }
        output_to_file += fmt::format( "# End: Data {}\n", datatype_out );
        output_to_file += fmt::format( "# End: Segment\n" );

        // Append the #End keywords
        if( append )
        {
            file_handle handle(file->file_name, true);
            handle.write( {output_to_file} );
        }
        else
        {
            file_handle handle(file->file_name, false);
            file->n_segments = 0;
            file->version = 2;
            handle.write( {top_header_string(), output_to_file} );
        }
        file->found  = true;
        file->is_ovf = true;

        // Increment the n_segments after succesfully appending the segment body to the file
        return increment_n_segments(file);
    }
    catch( const std::exception & ex )
    {
        file->_state->message_latest = fmt::format("Caught std::exception \"{}\"", ex.what());
        return OVF_ERROR;
    }
    catch( ... )
    {
        return OVF_ERROR;
    }
}
}
}

#endif