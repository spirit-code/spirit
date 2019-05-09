#pragma once
#ifndef LIBOVF_DETAIL_PARSE_H
#define LIBOVF_DETAIL_PARSE_H

#include "ovf.h"
#include <detail/parse_rules.hpp>

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <string>
#include <vector>
#include <array>
#include <fstream>
#include <iostream>

namespace ovf
{
namespace detail
{
namespace parse
{
    /*
    The number of zero-padding for segment count.
    This is needed so that, when appending, the file does not need to be overwritten.
    6: can store 1M segments
    */
    static const int n_segments_str_digits = 6;

    /*
    Parse the overall file header and locate the segment count, if present.
    */
    inline int file_header(ovf_file & file)
    try
    {
        pegtl::file_input<> in( file.file_name );
        bool success = pegtl::parse< ovf_file_header, ovf_file_action >( in, file );
        if( success )
            return OVF_OK;
        else
            return OVF_INVALID;
    }
    catch( pegtl::parse_error err )
    {
        file._state->message_latest = fmt::format(
            "libovf initial: pegtl parse error \'{}\'", + err.what());
        return OVF_ERROR;
    }
    catch( std::exception ex )
    {
        file._state->message_latest = fmt::format(
            "libovf initial: std::exception \'{}\'", + ex.what());
        return OVF_ERROR;
    }
    catch( ... )
    {
        file._state->message_latest = "libovf initial: unknown exception";
        return OVF_ERROR;
    }

    /*
    Read the overall file header and locate and count segments in the file
    (the segments are stored in memory!)
    */
    inline int initial(ovf_file & file)
    try
    {
        pegtl::file_input<> in( file.file_name );
        bool success = pegtl::parse< ovf_file_header, ovf_file_action >( in, file );
        if( success )
        {
            success = false;
            if( file.version == 2 )
            {
                success = pegtl::parse< pegtl::until<pegtl::until<pegtl::at< pegtl::seq<v2::begin, TAO_PEGTL_ISTRING("Segment"), pegtl::eol >>>> >( in, file );
                success = pegtl::parse< pegtl::plus<v2::segment>, v2::ovf_segment_action >( in, file );
            }
            else if( file.version == 1 )
            {
                // TODO...
                file._state->message_latest = fmt::format(
                    "libovf initial: OVF version \'{}\' in file \'{}\' is not supported...",
                    file.file_name, file.version);
            }
            else
            {
                file._state->message_latest = fmt::format(
                    "libovf initial: OVF version \'{}\' in file \'{}\' is not supported...",
                    file.file_name, file.version);
                return OVF_INVALID;
            }

            if( success )
            {
                int n_located = file._state->file_contents.size();
                if( file.n_segments != n_located )
                {
                    file._state->message_latest = fmt::format(
                        "libovf initial: n_segments specified in header ({}) is different from the number"
                        " of segments ({}) found in the file \'{}\'...",
                        file.n_segments, n_located, file.file_name);
                    return OVF_INVALID;
                }

                file.is_ovf = true;
                return OVF_OK;
            }
            else
            {
                file._state->message_latest = "libovf initial: no success in parsing...";
                return OVF_INVALID;
            }
        }
        else
        {
            file._state->message_latest = "libovf initial: no success in parsing...";
            return OVF_INVALID;
        }
    }
    catch( pegtl::parse_error err )
    {
        file._state->message_latest = fmt::format(
            "libovf initial: pegtl parse error \'{}\'", + err.what());
        return OVF_ERROR;
    }
    catch( std::exception ex )
    {
        file._state->message_latest = fmt::format(
            "libovf initial: std::exception \'{}\'", + ex.what());
        return OVF_ERROR;
    }
    catch( ... )
    {
        file._state->message_latest = "libovf initial: unknown exception";
        return OVF_ERROR;
    }

    // Reads in the header info into a given segment
    inline int segment_header(ovf_file & file, int index, ovf_segment & segment)
    try
    {
        pegtl::memory_input<> in( file._state->file_contents[index], "" );
        file._state->found_title        = false;
        file._state->found_meshunit     = false;
        file._state->found_valuedim     = false;
        file._state->found_valueunits   = false;
        file._state->found_valuelabels  = false;
        file._state->found_xmin         = false;
        file._state->found_ymin         = false;
        file._state->found_zmin         = false;
        file._state->found_xmax         = false;
        file._state->found_ymax         = false;
        file._state->found_zmax         = false;
        file._state->found_meshtype     = false;
        file._state->found_xbase        = false;
        file._state->found_ybase        = false;
        file._state->found_zbase        = false;
        file._state->found_xstepsize    = false;
        file._state->found_ystepsize    = false;
        file._state->found_zstepsize    = false;
        file._state->found_xnodes       = false;
        file._state->found_ynodes       = false;
        file._state->found_znodes       = false;
        file._state->found_pointcount   = false;
        bool success = false;

        if( file.version == 2 )
        {
            success = pegtl::parse< pegtl::plus<v2::segment_header>, v2::ovf_segment_header_action, v2::ovf_segment_header_control >( in, file, segment );
        }
        else if( file.version == 1 )
        {
            // TODO...
            file._state->message_latest = fmt::format(
                "libovf segment_header: OVF version \'{}\' in file \'{}\' is not supported...",
                file.file_name, file.version);
            return OVF_INVALID;
        }
        else
        {
            file._state->message_latest = fmt::format(
                "libovf segment_header: OVF version \'{}\' in file \'{}\' is not supported...",
                file.file_name, file.version);
            return OVF_INVALID;
        }

        if( success )
            return OVF_OK;
        else
        {
            file._state->message_latest = "libovf segment_header: no success in parsing";
            std::cerr << file._state->file_contents[index] << std::endl;
            return OVF_INVALID;
        }
    }
    catch( v2::keyword_value_line_error & err )
    {
        pegtl::memory_input<> in( file._state->file_contents[index], "" );
        const auto p = err.positions.front();
        std::string line = in.line_as_string(p);
        file._state->message_latest = fmt::format(
            "libovf segment_header: Expected an empty line or a line containing a keyword and a value!"
            "Found the following line instead:\n\"{}\"", line);
        return OVF_ERROR;
    }
    catch( pegtl::parse_error & err )
    {
        file._state->message_latest = fmt::format(
            "libovf segment_header: pegtl parse error \'{}\'", + err.what());
        return OVF_ERROR;
    }
    catch( std::exception ex )
    {
        file._state->message_latest = fmt::format(
            "libovf segment_header: std::exception \'{}\'", + ex.what());
        return OVF_ERROR;
    }
    catch( ... )
    {
        file._state->message_latest = "libovf segment_header: unknown exception";
        return OVF_ERROR;
    }

    // Reads the data of a segment into a given data array (float)
    template<typename scalar>
    int segment_data(ovf_file & file, int index, const ovf_segment & segment, scalar * data)
    try
    {
        pegtl::memory_input<> in( file._state->file_contents[index], "" );
        int retcode = OVF_ERROR;
        bool success = false;

        if( file.version == 2 )
        {
            file._state->max_data_index = segment.N*segment.valuedim;
            success = pegtl::parse< v2::segment_data, v2::ovf_segment_data_action >( in, file, segment, data );
            file._state->current_line = 0;
            file._state->current_column = 0;
            file._state->bin_data_idx = 0;
        }
        else if( file.version == 1 )
        {
            // TODO...
            file._state->message_latest = fmt::format(
                "libovf segment_data: OVF version \'{}\' in file \'{}\' is not supported...",
                file.file_name, file.version);
            return OVF_INVALID;
        }
        else
        {
            file._state->message_latest = fmt::format(
                "libovf segment_data: OVF version \'{}\' in file \'{}\' is not supported...",
                file.file_name, file.version);
            return OVF_INVALID;
        }

        if( success )
        {
            return OVF_OK;
        }
        else
        {
            file._state->message_latest = "libovf segment_data: no success in parsing";
            return OVF_INVALID;
        }
    }
    catch( max_index_error )
    {
        file._state->current_line = 0;
        file._state->current_column = 0;
        file._state->bin_data_idx = 0;
        return OVF_OK;
    }
    catch( pegtl::parse_error err )
    {
        file._state->message_latest = fmt::format(
            "libovf segment_data: pegtl parse error \'{}\'", + err.what());
        return OVF_ERROR;
    }
    catch( std::exception ex )
    {
        file._state->message_latest = fmt::format(
            "libovf segment_data: std::exception \'{}\'", + ex.what());
        return OVF_ERROR;
    }
    catch( ... )
    {
        file._state->message_latest = "libovf segment_data: unknown exception";
        return OVF_ERROR;
    }
}
}
}

#endif