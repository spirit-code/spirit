#pragma once
#ifndef LIBOVF_DETAIL_PARSE_RULES_H
#define LIBOVF_DETAIL_PARSE_RULES_H

#include "ovf.h"
#include <detail/helpers.hpp>

#include <tao/pegtl.hpp>
#include <fmt/format.h>

#include <array>

struct max_index_error : public std::runtime_error
{
    max_index_error() : std::runtime_error("") {};
};

struct parser_state
{
    // For the segment strings
    std::vector<std::string> file_contents{};

    // for reading data blocks
    int current_column = 0;
    int current_line = 0;
    int bin_data_idx = 0;

    std::string keyword="", value="";

    // Whether certain keywords were found in parsing
    bool found_title        = false;
    bool found_meshunit     = false;
    bool found_valuedim     = false;
    bool found_valueunits   = false;
    bool found_valuelabels  = false;
    bool found_xmin         = false;
    bool found_ymin         = false;
    bool found_zmin         = false;
    bool found_xmax         = false;
    bool found_ymax         = false;
    bool found_zmax         = false;
    bool found_meshtype     = false;
    bool found_xbase        = false;
    bool found_ybase        = false;
    bool found_zbase        = false;
    bool found_xstepsize    = false;
    bool found_ystepsize    = false;
    bool found_zstepsize    = false;
    bool found_xnodes       = false;
    bool found_ynodes       = false;
    bool found_znodes       = false;
    bool found_pointcount   = false;

    /*
    messages, e.g. in case a function returned OVF_ERROR.
    message_out will be filled and returned by ovf_latest_message, while message_latest
    will be filled by other functions and cleared by ovf_latest_message.
    */
    std::string message_out="", message_latest="";

    int max_data_index=0;
    int tmp_idx=0;
    std::array<double, 3> tmp_vec3 = std::array<double, 3>{0,0,0};

    std::ios::pos_type n_segments_pos = 0;
};

namespace ovf
{
namespace detail
{
namespace parse
{
    namespace pegtl = tao::pegtl;

    // "# "
    struct prefix
        : pegtl::string< '#' >
    {};

    // "#\eol"
    struct empty_line
        : pegtl::seq< pegtl::string< '#' >, pegtl::star<pegtl::blank> >
    {};

    //
    struct version_number
        : pegtl::range< '1', '2' >
    {};

    // " OOMMF OVF "
    struct version
        : pegtl::seq< prefix, pegtl::pad< TAO_PEGTL_ISTRING("OOMMF OVF"), pegtl::blank >, version_number, pegtl::until<pegtl::eol> >
    {};

    // " Segment count: "
    struct segment_count_number
        : pegtl::plus<pegtl::digit>
    {};

    // " Segment count: "
    struct segment_count
        : pegtl::seq< prefix, pegtl::pad< TAO_PEGTL_ISTRING("Segment count:"), pegtl::blank >, segment_count_number, pegtl::eol >
    {};

    //////////////////////////

    struct ovf_file_header
        : pegtl::must<
            version,
            pegtl::star<pegtl::seq<empty_line, pegtl::eol>>,
            segment_count>
    {};

    //////////////////////////

    // Class template for user-defined actions that does nothing by default.
    template< typename Rule >
    struct ovf_file_action
        : pegtl::nothing< Rule >
    {};

    template<>
    struct ovf_file_action< version_number >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & file )
        {
            file.version = std::stoi(in.string());
        }
    };

    template<>
    struct ovf_file_action< segment_count_number >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & file )
        {
            file.n_segments = std::stoi(in.string());
            file._state->n_segments_pos = in.position().byte;
        }
    };

    //////////////////////////

    namespace v2
    {
        // "# "
        struct prefix
            : pegtl::string< '#' >
        {};

        // Line without contents, up to EOL or comment
        struct empty_line
            : pegtl::seq<
                pegtl::string<'#'>,
                pegtl::until<
                    pegtl::at<
                        pegtl::sor<pegtl::eol, pegtl::string<'#','#'>>
                    >,
                    pegtl::blank
                >
            >
        {};

        // Comment line up to EOL. "##" initiates comment line
        struct comment
            : pegtl::seq<
                pegtl::string< '#', '#' >,
                pegtl::until<
                    pegtl::at<pegtl::eol>,
                    pegtl::any
                >
            >
        {};

        // Number of lines without content (empty and comment lines)
        struct skippable_lines
            : pegtl::star< pegtl::sor<
                pegtl::seq< empty_line, pegtl::opt<comment>, pegtl::eol >,
                pegtl::seq< comment, pegtl::eol >
            > >
        {};

        // " OOMMF OVF "
        struct version
            : pegtl::seq< prefix, pegtl::pad< TAO_PEGTL_ISTRING("OOMMF OVF"), pegtl::blank >, pegtl::range< '1', '2' >, pegtl::until<pegtl::eol> >
        {};

        // " Segment count: "
        struct segment_count_number
            : pegtl::plus<pegtl::digit>
        {};

        // " Segment count: "
        struct segment_count
            : pegtl::seq< prefix, pegtl::pad< TAO_PEGTL_ISTRING("Segment count:"), pegtl::blank >, segment_count_number, pegtl::eol >
        {};

        // " Begin: "
        struct begin
            : pegtl::seq< prefix, pegtl::pad< TAO_PEGTL_ISTRING("Begin:"), pegtl::blank > >
        {};

        // " End: "
        struct end
            : pegtl::seq< prefix, pegtl::pad< TAO_PEGTL_ISTRING("End:"), pegtl::blank > >
        {};

        //////////////////////////////////////////////

        struct opt_plus_minus
            : pegtl::opt< pegtl::one< '+', '-' > >
        {};

        struct inf
            : pegtl::seq<
                pegtl::istring< 'i', 'n', 'f' >,
                pegtl::opt< pegtl::istring< 'i', 'n', 'i', 't', 'y' > > >
        {};

        struct nan
            : pegtl::seq<
                pegtl::istring< 'n', 'a', 'n' >,
                pegtl::opt< pegtl::one< '(' >,
                            pegtl::plus< pegtl::alnum >,
                            pegtl::one< ')' > > >
        {};

        template< typename D >
        struct basic_number
            : pegtl::if_then_else<
                pegtl::one< '.' >,
                pegtl::plus< D >,
                pegtl::seq<
                    pegtl::plus< D >,
                    pegtl::opt< pegtl::one< '.' > >,
                    pegtl::star< D >
                >
            >
        {};

        struct exponent
            : pegtl::seq<
                opt_plus_minus,
                pegtl::plus< pegtl::digit > >
        {};

        struct decimal_number
            : pegtl::seq<
                basic_number< pegtl::digit >,
                pegtl::opt< pegtl::one< 'e', 'E' >, exponent > >
        {};

        struct hexadecimal_number // TODO: is this actually hexadecimal??
            : pegtl::seq<
                pegtl::one< '0' >,
                pegtl::one< 'x', 'X' >,
                basic_number< pegtl::xdigit >,
                pegtl::opt< pegtl::one< 'p', 'P' >, exponent > >
        {};

        struct float_value
            : pegtl::seq<
                opt_plus_minus,
                decimal_number >
        {};

        struct vec_float_value
            : pegtl::seq<
                opt_plus_minus,
                decimal_number >
        {};


        struct data_float
            : pegtl::seq<
                opt_plus_minus,
                decimal_number >
        {};

        struct segment_data_float
            : data_float
        {};
        struct line_data_txt
            : pegtl::plus< pegtl::pad< segment_data_float, pegtl::blank > >
        {};
        struct line_data_csv
            : pegtl::seq<
                pegtl::list< pegtl::pad<segment_data_float, pegtl::blank>, pegtl::one<','> >,
                pegtl::opt< pegtl::pad< pegtl::one<','>, pegtl::blank > >
                >
        {};

        struct bin_4_check_value
            : tao::pegtl::uint32_le::any
        {};
        struct bin_4_value
            : tao::pegtl::uint32_le::any
        {};

        struct bin_8_check_value
            : tao::pegtl::uint64_le::any
        {};
        struct bin_8_value
            : tao::pegtl::uint64_le::any
        {};

        //////////////////////////////////////////////

        // Vector3 of floating point values
        struct vector3f
            : pegtl::rep< 3, pegtl::pad< vec_float_value, pegtl::blank > >
        {};

        // This is how a line ends: either eol or the begin of a comment
        struct line_end
            : pegtl::sor<pegtl::eol, pegtl::string<'#','#'>>
        {};

        // This checks that the line end is met and moves up until eol
        struct finish_line
            : pegtl::seq< pegtl::star<pegtl::blank>, pegtl::at<line_end>, pegtl::until<pegtl::eol>>
        {};

        //////////////////////////////////////////////

        struct keyword
            : pegtl::until< pegtl::at<TAO_PEGTL_ISTRING(":")>,
                pegtl::if_must<pegtl::not_at<pegtl::eol>, pegtl::any> >//, pegtl::not_at<pegtl::sor<pegtl::eol, TAO_PEGTL_ISTRING("##")>> >
        {};

        struct value   : pegtl::seq< pegtl::until<pegtl::at< line_end >> > {};

        struct keyword_value_line
            : pegtl::seq<
                prefix,
                pegtl::pad< keyword, pegtl::blank >,
                TAO_PEGTL_ISTRING(":"),
                pegtl::pad< value, pegtl::blank >,
                finish_line >
        {};

        //
        struct header
            : pegtl::seq<
                begin, TAO_PEGTL_ISTRING("Header"), finish_line,
                pegtl::until<
                    pegtl::seq<end, TAO_PEGTL_ISTRING("Header")>,
                    pegtl::must<
                        skippable_lines,
                        keyword_value_line,
                        skippable_lines
                    >
                >,
                finish_line
            >
        {};

        //
        struct segment
            : pegtl::seq<
                pegtl::star<pegtl::seq<empty_line, pegtl::eol>>,
                pegtl::seq< begin, TAO_PEGTL_ISTRING("Segment"), pegtl::eol>,
                // pegtl::star<pegtl::seq<empty_line, pegtl::eol>>,
                // header,
                // pegtl::star<pegtl::seq<empty_line, pegtl::eol>>,
                // pegtl::sor<data_text, data_csv, data_binary_8, data_binary_4>,
                // pegtl::star<pegtl::seq<empty_line, pegtl::eol>>,
                pegtl::until<pegtl::seq<end, TAO_PEGTL_ISTRING("Segment")>>, pegtl::eol >
        {};

        // Class template for user-defined actions that does nothing by default.
        template< typename Rule >
        struct ovf_segment_action
            : pegtl::nothing< Rule >
        {};

        template<>
        struct ovf_segment_action< segment >
        {
            template< typename Input >
            static void apply( const Input& in, ovf_file & file )
            {
                file._state->file_contents.push_back(in.string());
            }
        };

        //////////////////////////////////////////////

        //
        struct segment_header
            : pegtl::seq<
                pegtl::star<pegtl::seq<empty_line, pegtl::eol>>,
                pegtl::seq< begin, TAO_PEGTL_ISTRING("Segment"), pegtl::eol>,
                pegtl::star<pegtl::seq<empty_line, pegtl::eol>>,
                header,
                pegtl::star<pegtl::seq<empty_line, pegtl::eol>>,
                // pegtl::sor<data_text, data_csv, data_binary_8, data_binary_4>,
                // pegtl::star<pegtl::seq<empty_line, pegtl::eol>>,
                pegtl::until<pegtl::seq<end, TAO_PEGTL_ISTRING("Segment")>>, pegtl::eol >
        {};

        // Class template for user-defined actions that does nothing by default.
        template< typename Rule >
        struct ovf_segment_header_action
            : pegtl::nothing< Rule >
        {};

        template<>
        struct ovf_segment_header_action< segment_header >
        {
            template< typename Input >
            static void apply( const Input& in, ovf_file & file, ovf_segment & segment )
            {
                // Check if all required keywords were present
                std::vector<std::string> missing_keywords(0);
                if( !file._state->found_title )
                    missing_keywords.push_back("title");
                if( !file._state->found_meshunit )
                    missing_keywords.push_back("meshunit");
                if( !file._state->found_valueunits )
                    missing_keywords.push_back("valueunits");
                if( !file._state->found_valuelabels )
                    missing_keywords.push_back("valuelabels");
                if( !file._state->found_xmin )
                    missing_keywords.push_back("xmin");
                if( !file._state->found_ymin )
                    missing_keywords.push_back("ymin");
                if( !file._state->found_zmin )
                    missing_keywords.push_back("zmin");
                if( !file._state->found_xmax )
                    missing_keywords.push_back("xmax");
                if( !file._state->found_ymax )
                    missing_keywords.push_back("ymax");
                if( !file._state->found_zmax )
                    missing_keywords.push_back("zmax");
                if( !file._state->found_meshtype )
                    missing_keywords.push_back("meshtype");

                if( std::string(segment.meshtype) == "rectangular" )
                {
                    segment.N = segment.n_cells[0] * segment.n_cells[1] * segment.n_cells[2];

                    if( !file._state->found_xbase )
                        missing_keywords.push_back("xbase");
                    if( !file._state->found_ybase )
                        missing_keywords.push_back("ybase");
                    if( !file._state->found_zbase )
                        missing_keywords.push_back("zbase");
                    if( !file._state->found_xstepsize )
                        missing_keywords.push_back("xstepsize");
                    if( !file._state->found_ystepsize )
                        missing_keywords.push_back("ystepsize");
                    if( !file._state->found_zstepsize )
                        missing_keywords.push_back("zstepsize");
                    if( !file._state->found_xnodes )
                        missing_keywords.push_back("xnodes");
                    if( !file._state->found_ynodes )
                        missing_keywords.push_back("ynodes");
                    if( !file._state->found_znodes )
                        missing_keywords.push_back("znodes");
                }
                else if( std::string(segment.meshtype) == "irregular" )
                {
                    segment.N = segment.pointcount;

                    if( !file._state->found_pointcount )
                        missing_keywords.push_back("pointcount");
                }

                if( missing_keywords.size() > 0 )
                {
                    std::string message = fmt::format( "Missing keywords: \"{}\"", missing_keywords[0] );
                    for( int i=1; i < missing_keywords.size(); ++i )
                        message += fmt::format( ", \"{}\"", missing_keywords[i] );
                    throw tao::pegtl::parse_error( message, in );
                }
            }
        };


        template<>
        struct ovf_segment_header_action< vec_float_value >
        {
            template< typename Input >
            static void apply( const Input& in, ovf_file & file, ovf_segment & segment )
            {
                file._state->tmp_vec3[file._state->tmp_idx] = std::stod(in.string());
                ++file._state->tmp_idx;
            }
        };

        ////////////////////////////////////////////////////////////////////////////////////////////

        template<>
        struct ovf_segment_header_action< keyword >
        {
            template< typename Input >
            static void apply( const Input& in, ovf_file & f, ovf_segment & segment )
            {
                f._state->keyword = in.string();
                std::transform(f._state->keyword.begin(), f._state->keyword.end(),f._state->keyword.begin(), ::tolower);
            }
        };

        template<>
        struct ovf_segment_header_action< value >
        {
            template< typename Input >
            static void apply( const Input& in, ovf_file & f, ovf_segment & segment )
            {
                f._state->value = in.string();
            }
        };

        template<>
        struct ovf_segment_header_action< keyword_value_line >
        {
            template< typename Input >
            static void apply( const Input& in, ovf_file & f, ovf_segment & segment )
            {
                if( f._state->keyword == "title" )
                {
                    segment.title = strdup(f._state->value.c_str());
                    f._state->found_title = true;
                }
                else if( f._state->keyword == "desc" )
                    segment.comment = strdup(f._state->value.c_str());
                else if( f._state->keyword == "meshunit" )
                {
                    segment.meshunit = strdup(f._state->value.c_str());
                    f._state->found_meshunit = true;
                }
                else if( f._state->keyword == "valuedim" )
                {
                    segment.valuedim = std::stoi(f._state->value.c_str());
                    f._state->found_valuedim = true;
                }
                else if( f._state->keyword == "valueunits" )
                {
                    segment.valueunits = strdup(f._state->value.c_str());
                    f._state->found_valueunits = true;
                }
                else if( f._state->keyword == "valuelabels" )
                {
                    segment.valuelabels = strdup(f._state->value.c_str());
                    f._state->found_valuelabels = true;
                }
                else if( f._state->keyword == "xmin" )
                {
                    segment.bounds_min[0] = std::stof(f._state->value.c_str());
                    f._state->found_xmin = true;
                }
                else if( f._state->keyword == "ymin" )
                {
                    segment.bounds_min[1] = std::stof(f._state->value.c_str());
                    f._state->found_ymin = true;
                }
                else if( f._state->keyword == "zmin" )
                {
                    segment.bounds_min[2] = std::stof(f._state->value.c_str());
                    f._state->found_zmin = true;
                }
                else if( f._state->keyword == "xmax" )
                {
                    segment.bounds_max[0] = std::stof(f._state->value.c_str());
                    f._state->found_xmax = true;
                }
                else if( f._state->keyword == "ymax" )
                {
                    segment.bounds_max[1] = std::stof(f._state->value.c_str());
                    f._state->found_ymax = true;
                }
                else if( f._state->keyword == "zmax" )
                {
                    segment.bounds_max[2] = std::stof(f._state->value.c_str());
                    f._state->found_zmax = true;
                }
                else if( f._state->keyword == "meshtype" )
                {
                    std::string meshtype = f._state->value;
                    std::transform(meshtype.begin(), meshtype.end(), meshtype.begin(), ::tolower);
                    if( std::string(segment.meshtype) == "" )
                    {
                        if( meshtype != "rectangular" && meshtype != "irregular" )
                            throw tao::pegtl::parse_error( fmt::format(
                                "Invalid meshtype: \"{}\"", meshtype), in );
                        segment.meshtype = strdup(meshtype.c_str());
                    }
                    else if( std::string(segment.meshtype) != meshtype )
                    {
                        throw tao::pegtl::parse_error( fmt::format(
                            "meshtype \"{}\" was specified, but due to other parameters specified before, \"{}\" was expected!",
                            meshtype, segment.meshtype), in );
                    }
                    f._state->found_meshtype = true;
                }
                else if( f._state->keyword == "xbase" )
                {
                    if( std::string(segment.meshtype) != "rectangular" )
                        throw tao::pegtl::parse_error( fmt::format(
                            "xbase is only for rectangular meshes! Mesh type is \"{}\"", segment.meshtype), in );
                    segment.meshtype = strdup("rectangular");
                    segment.origin[0] = std::stof(f._state->value.c_str());
                    f._state->found_xbase = true;
                }
                else if( f._state->keyword == "ybase" )
                {
                    if( std::string(segment.meshtype) != "rectangular" )
                        throw tao::pegtl::parse_error( fmt::format(
                            "ybase is only for rectangular meshes! Mesh type is \"{}\"", segment.meshtype), in );
                    segment.meshtype = strdup("rectangular");
                    segment.origin[1] = std::stof(f._state->value.c_str());
                    f._state->found_ybase = true;
                }
                else if( f._state->keyword == "zbase" )
                {
                    if( std::string(segment.meshtype) != "rectangular" )
                        throw tao::pegtl::parse_error( fmt::format(
                            "zbase is only for rectangular meshes! Mesh type is \"{}\"", segment.meshtype), in );
                    segment.meshtype = strdup("rectangular");
                    segment.origin[2] = std::stof(f._state->value.c_str());
                    f._state->found_zbase = true;
                }
                else if( f._state->keyword == "xstepsize" )
                {
                    if( std::string(segment.meshtype) != "rectangular" )
                        throw tao::pegtl::parse_error( fmt::format(
                            "xstepsize is only for rectangular meshes! Mesh type is \"{}\"", segment.meshtype), in );
                    segment.meshtype = strdup("rectangular");
                    segment.step_size[0] = std::stof(f._state->value.c_str());
                    f._state->found_xstepsize = true;
                }
                else if( f._state->keyword == "ystepsize" )
                {
                    if( std::string(segment.meshtype) != "rectangular" )
                        throw tao::pegtl::parse_error( fmt::format(
                            "ystepsize is only for rectangular meshes! Mesh type is \"{}\"", segment.meshtype), in );
                    segment.meshtype = strdup("rectangular");
                    segment.step_size[1] = std::stof(f._state->value.c_str());
                    f._state->found_ystepsize = true;
                }
                else if( f._state->keyword == "zstepsize" )
                {
                    if( std::string(segment.meshtype) != "rectangular" )
                        throw tao::pegtl::parse_error( fmt::format(
                            "zstepsize is only for rectangular meshes! Mesh type is \"{}\"", segment.meshtype), in );
                    segment.meshtype = strdup("rectangular");
                    segment.step_size[2] = std::stof(f._state->value.c_str());
                    f._state->found_zstepsize = true;
                }
                else if( f._state->keyword == "xnodes" )
                {
                    if( std::string(segment.meshtype) != "rectangular" )
                        throw tao::pegtl::parse_error( fmt::format(
                            "xnodes is only for rectangular meshes! Mesh type is \"{}\"", segment.meshtype), in );
                    segment.meshtype = strdup("rectangular");
                    segment.n_cells[0] = std::stoi(f._state->value.c_str());
                    f._state->found_xnodes = true;
                }
                else if( f._state->keyword == "ynodes" )
                {
                    if( std::string(segment.meshtype) != "rectangular" )
                        throw tao::pegtl::parse_error( fmt::format(
                            "ynodes is only for rectangular meshes! Mesh type is \"{}\"", segment.meshtype), in );
                    segment.meshtype = strdup("rectangular");
                    segment.n_cells[1] = std::stoi(f._state->value.c_str());
                    f._state->found_ynodes = true;
                }
                else if( f._state->keyword == "znodes" )
                {
                    if( std::string(segment.meshtype) != "rectangular" )
                        throw tao::pegtl::parse_error( fmt::format(
                            "znodes is only for rectangular meshes! Mesh type is \"{}\"", segment.meshtype), in );
                    segment.meshtype = strdup("rectangular");
                    segment.n_cells[2] = std::stoi(f._state->value.c_str());
                    f._state->found_znodes = true;
                }
                else if( f._state->keyword == "pointcount" )
                {
                    if( std::string(segment.meshtype) != "" && std::string(segment.meshtype) != "irregular" )
                        throw tao::pegtl::parse_error( fmt::format(
                            "pointcount is only for irregular meshes! Mesh type is \"{}\"", segment.meshtype), in );
                    segment.meshtype = strdup("irregular");
                    segment.pointcount = std::stoi(f._state->value.c_str());
                    f._state->found_pointcount = true;
                }
                else
                {
                    // UNKNOWN KEYWORD
                    throw tao::pegtl::parse_error( fmt::format(
                        "unknown keyword \"{}\": \"{}\"", f._state->keyword, f._state->value), in );
                }

                f._state->keyword = "";
                f._state->value = "";
            }
        };

        ////////////////////////////////////////////////////////////////////////////////////////////

        struct data_text
            : pegtl::seq<
                begin, TAO_PEGTL_ISTRING("Data Text"), pegtl::eol,
                pegtl::plus< line_data_txt, pegtl::eol >,
                end, TAO_PEGTL_ISTRING("Data Text"), pegtl::eol
                >
        {};

        struct data_csv
            : pegtl::seq<
                begin, TAO_PEGTL_ISTRING("Data CSV"), pegtl::eol,
                pegtl::plus< line_data_csv, pegtl::eol >,
                end, TAO_PEGTL_ISTRING("Data CSV"), pegtl::eol
                >
        {};

        struct data_binary_4
            : pegtl::seq<
                begin, TAO_PEGTL_ISTRING("Data Binary 4"), pegtl::eol,
                bin_4_check_value,
                pegtl::until< pegtl::seq<end, TAO_PEGTL_ISTRING("Data Binary 4"), pegtl::eol>, bin_4_value >
                >
        {};

        struct data_binary_8
            : pegtl::seq<
                begin, TAO_PEGTL_ISTRING("Data Binary 8"), pegtl::eol,
                bin_8_check_value,
                pegtl::until< pegtl::seq<end, TAO_PEGTL_ISTRING("Data Binary 8"), pegtl::eol>, bin_8_value >
                >
        {};

        struct segment_data
            : pegtl::seq<
                pegtl::star<pegtl::seq<empty_line, pegtl::eol>>,
                begin, TAO_PEGTL_ISTRING("Segment"), pegtl::eol,
                pegtl::star<pegtl::seq<empty_line, pegtl::eol>>,
                header,
                pegtl::star<pegtl::seq<empty_line, pegtl::eol>>,
                pegtl::sor< data_text, data_csv, data_binary_4, data_binary_8 >,
                pegtl::star<pegtl::seq<empty_line, pegtl::eol>>,
                pegtl::until<pegtl::seq<end, TAO_PEGTL_ISTRING("Segment")>>, pegtl::eol >
        {};

        ////////////////////////////////////

        // Class template for user-defined actions that does nothing by default.
        template< typename Rule >
        struct ovf_segment_data_action
            : pegtl::nothing< Rule >
        {};

        // template<>
        // struct ovf_segment_data_action< data_text >
        // {
        //     template< typename Input, typename scalar >
        //     static void apply( const Input& in, ovf_file & f, const ovf_segment & segment, scalar * data )
        //     {
        //         std::cerr << "\nsegment data action triggered\n" << std::endl;
        //     }
        // };

        template<>
        struct ovf_segment_data_action< line_data_txt >
        {
            template< typename Input, typename scalar >
            static void apply( const Input& in, ovf_file & f, const ovf_segment & segment, scalar * data )
            {
                f._state->current_column = 0;
                ++f._state->current_line;
            }
        };

        template<>
        struct ovf_segment_data_action< line_data_csv >
        {
            template< typename Input, typename scalar >
            static void apply( const Input& in, ovf_file & f, const ovf_segment & segment, scalar * data )
            {
                f._state->current_column = 0;
                ++f._state->current_line;
            }
        };

        template<>
        struct ovf_segment_data_action< segment_data_float >
        {
            template< typename Input, typename scalar >
            static void apply( const Input& in, ovf_file & f, const ovf_segment & segment, scalar * data )
            {
                int row = f._state->current_line;
                int col = f._state->current_column;

                int n_cols = segment.valuedim;

                double value = std::stod(in.string());

                int idx = col + row*n_cols;

                if( idx < f._state->max_data_index )
                {
                    data[idx] = value;
                    ++f._state->current_column;
                }
                else
                    throw max_index_error();
            }
        };

        template<>
        struct ovf_segment_data_action< bin_4_check_value >
        {
            template< typename Input, typename scalar >
            static void apply( const Input& in, ovf_file & f, const ovf_segment & segment, scalar * data )
            {
                std::string bytes = in.string();
                uint32_t hex_4b = endian::from_little_32(reinterpret_cast<const uint8_t *>( bytes.c_str() ));

                if ( hex_4b != check::val_4b )
                    throw tao::pegtl::parse_error( "the expected binary check value could not be parsed!", in );
            }
        };

        template<>
        struct ovf_segment_data_action< bin_4_value >
        {
            template< typename Input, typename scalar >
            static void apply( const Input& in, ovf_file & f, const ovf_segment & segment, scalar * data )
            {
                std::string bytes = in.string();
                uint32_t ivalue = endian::from_little_32(reinterpret_cast<const uint8_t *>( bytes.c_str() ));
                float value = *reinterpret_cast<const float *>( &ivalue );

                int row = f._state->current_line;
                int col = f._state->current_column;

                int n_cols = segment.valuedim;

                int idx = col + row*n_cols;

                if( idx < f._state->max_data_index )
                {
                    data[idx] = value;
                    ++f._state->current_column;
                }
                else
                    throw max_index_error();
            }
        };

        template<>
        struct ovf_segment_data_action< bin_8_check_value >
        {
            template< typename Input, typename scalar >
            static void apply( const Input& in, ovf_file & f, const ovf_segment & segment, scalar * data )
            {
                std::string bytes = in.string();
                uint64_t hex_8b = endian::from_little_64(reinterpret_cast<const uint8_t *>( bytes.c_str() ));

                if ( hex_8b != check::val_8b )
                    throw tao::pegtl::parse_error( "the expected binary check value could not be parsed!", in );
            }
        };

        template<>
        struct ovf_segment_data_action< bin_8_value >
        {
            template< typename Input, typename scalar >
            static void apply( const Input& in, ovf_file & f, const ovf_segment & segment, scalar * data )
            {
                std::string bytes = in.string();
                uint64_t ivalue = endian::from_little_64(reinterpret_cast<const uint8_t *>( bytes.c_str() ));
                double value = *reinterpret_cast<const double *>( &ivalue );

                int row = f._state->current_line;
                int col = f._state->current_column;

                int n_cols = segment.valuedim;

                int idx = f._state->bin_data_idx;
                ++f._state->bin_data_idx;

                if( idx < f._state->max_data_index )
                {
                    data[idx] = value;
                    ++f._state->current_column;
                }
                else
                    throw max_index_error();
            }
        };

        template<>
        struct ovf_segment_data_action< data_binary_4 >
        {
            template< typename Input, typename scalar >
            static void apply( const Input& in, ovf_file & f, const ovf_segment & segment, scalar * data )
            {
                static_assert(
                    !std::is_floating_point<scalar>::value ||
                    (std::is_floating_point<scalar>::value && std::numeric_limits<scalar>::is_iec559),
                    "Portable binary only supports IEEE 754 standardized floating point" );
            }
        };

        template<>
        struct ovf_segment_data_action< data_binary_8 >
        {
            template< typename Input, typename scalar >
            static void apply( const Input& in, ovf_file & f, const ovf_segment & segment, scalar * data )
            {
                static_assert(
                    !std::is_floating_point<scalar>::value ||
                    (std::is_floating_point<scalar>::value && std::numeric_limits<scalar>::is_iec559),
                    "Portable binary only supports IEEE 754 standardized floating point" );
            }
        };

        //////////////////////////////////////////////

        template< typename Rule >
        struct ovf_segment_header_control : tao::pegtl::normal< Rule >
        {
            template< typename Input, typename... States >
            static void raise( const Input& in, States&&... );
        };

        template< typename T> template< typename Input, typename... States >
        void ovf_segment_header_control< T >::raise( const Input& in, States&&... )
        {
            throw tao::pegtl::parse_error( "parse error matching " + tao::pegtl::internal::demangle< T >(), in );
        }

        struct keyword_value_line_error : public tao::pegtl::parse_error
        {
            template< typename Input >
            keyword_value_line_error(const Input & in) : tao::pegtl::parse_error("", in) {};
        };

        template<> template< typename Input, typename... States >
        void ovf_segment_header_control< keyword_value_line >::raise( const Input& in, States&&... )
        {
            throw keyword_value_line_error( in );
        }
    }; // namespace v2

    namespace v1
    {
        struct file
            : pegtl::star<pegtl::any>
        {};

        // Class template for user-defined actions that does nothing by default.
        template< typename Rule >
        struct file_action
            : pegtl::nothing< Rule >
        {};
    };

} // namespace parse
}
}

#endif