#pragma once
#ifndef LIBOVF_DETAIL_PARSE_RULES_H
#define LIBOVF_DETAIL_PARSE_RULES_H

#include "ovf.h"
#include "keywords.hpp"
#include "atomistic_keywords.hpp"
#include "pegtl_defines.hpp"
#include <detail/helpers.hpp>

#include <tao/pegtl.hpp>
#include <fmt/format.h>

#include <array>


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

    // "%"
    struct magic_char
        : pegtl::string< '%' >
    {};

    // "##% "
    struct magic_prefix
        : pegtl::seq< pegtl::string< '#', '#'>, magic_char >
    {};

    // "#\eol"
    struct empty_line
        : pegtl::seq< pegtl::string< '#' >, pegtl::star<pegtl::blank> >
    {};

    //
    struct version_number
        : pegtl::range< '1', '2' >
    {};

    struct version_string
        : pegtl::sor< TAO_PEGTL_ISTRING("OOMMF OVF"), TAO_PEGTL_ISTRING("AOVF_COMP"), TAO_PEGTL_ISTRING("AOVF") >
    {};

    // " OOMMF OVF "
    struct version
        :
            pegtl::sor<
                pegtl::seq< prefix, pegtl::pad< TAO_PEGTL_ISTRING("OOMMF OVF"), pegtl::blank>, pegtl::until<pegtl::eol>, magic_prefix, pegtl::pad< version_string, pegtl::blank>, version_number, pegtl::until<pegtl::eol> >,
                pegtl::seq< prefix, pegtl::pad< version_string, pegtl::blank >, version_number, pegtl::until<pegtl::eol> >
            >
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
    struct ovf_file_action< version_string >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & file )
        {
            file.version_string = strdup(in.string().c_str());
            if(in.string() == "AOVF_COMP")
            {
                file.ovf_extension_format = OVF_EXTENSION_FORMAT_AOVF_COMP;
            } else if(in.string() == "AOVF")
            {
                file.ovf_extension_format = OVF_EXTENSION_FORMAT_AOVF;
            } else if(in.string() == "OOMMF OVF")
            {
                file.ovf_extension_format = OVF_EXTENSION_FORMAT_OVF;
            } else {
                throw pegtl::parse_error(fmt::format("Detected invalid version string {}", in.string()), in);
            }
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
                        pegtl::sor<pegtl::eol, pegtl::seq<pegtl::string<'#','#'>, pegtl::not_at<magic_char>>>
                    >,
                    pegtl::blank
                >
            >
        {};

        // Comment line up to EOL. "##" initiates comment line
        struct comment
            : pegtl::seq<
                pegtl::string< '#', '#' >,
                pegtl::not_at<magic_char>,
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


        struct text_value_float
            : pegtl::seq<
                opt_plus_minus,
                decimal_number >
        {};

        struct line_data_txt
            : pegtl::plus< pegtl::pad< text_value_float, pegtl::blank > >
        {};
        struct line_data_csv
            : pegtl::seq<
                pegtl::list< pegtl::pad<text_value_float, pegtl::blank>, pegtl::one<','> >,
                pegtl::opt< pegtl::pad< pegtl::one<','>, pegtl::blank > >
                >
        {};

        struct check_value_bin_4
            : tao::pegtl::uint32_le::any
        {};
        struct bytes_bin_4
            : pegtl::seq< pegtl::star< pegtl::not_at< pegtl::seq<end, TAO_PEGTL_ISTRING("Data Binary 4"), pegtl::eol> >, pegtl::any > >
        {};

        struct check_value_bin_8
            : tao::pegtl::uint64_le::any
        {};
        struct bytes_bin_8
            : pegtl::seq< pegtl::star< pegtl::not_at< pegtl::seq<end, TAO_PEGTL_ISTRING("Data Binary 8"), pegtl::eol> >, pegtl::any > >
        {};

        //////////////////////////////////////////////

        // Vector3 of floating point values
        struct vector3f
            : pegtl::rep< 3, pegtl::pad< vec_float_value, pegtl::blank > >
        {};

        // This is how a line ends: either eol or the begin of a comment
        struct line_end
            : pegtl::sor<pegtl::eol, pegtl::seq<pegtl::string<'#','#'>,pegtl::not_at<magic_char>>>
        {};

        // This checks that the line end is met and moves up until eol
        struct finish_line
            : pegtl::seq< pegtl::star<pegtl::blank>, pegtl::at<line_end>, pegtl::until<pegtl::eol>>
        {};

        //////////////////////////////////////////////

        template<typename kw, typename val>
        struct keyword_value_pair
            : pegtl::seq<
                prefix,
                pegtl::pad< kw, pegtl::blank >,
                TAO_PEGTL_ISTRING(":"),
                pegtl::pad< val, pegtl::blank >,
                pegtl::until<pegtl::at<line_end>>,
                finish_line >
        {};

        template<typename kw, typename val>
        struct magic_keyword_value_pair
            : pegtl::seq<
                magic_prefix,
                pegtl::pad< kw, pegtl::blank >,
                TAO_PEGTL_ISTRING(":"),
                pegtl::pad< val, pegtl::blank >,
                pegtl::until<pegtl::at<line_end>>,
                finish_line >
        {};

        struct ovf_keyword_value_line
            : pegtl::sor< 
                keyword_value_pair< keywords::title, keywords::title_value >,
                keyword_value_pair< keywords::desc, keywords::desc_value >,
                keyword_value_pair< keywords::valuedim, keywords::valuedim_value >,
                keyword_value_pair< keywords::valueunits, keywords::valueunits_value >,
                keyword_value_pair< keywords::valuelabels, keywords::valuelabels_value >,
                keyword_value_pair< keywords::meshtype, keywords::meshtype_value >,
                keyword_value_pair< keywords::meshunit, keywords::meshunit_value >,
                keyword_value_pair< keywords::pointcount, keywords::pointcount_value >,
                keyword_value_pair< keywords::xnodes, keywords::xnodes_value >,
                keyword_value_pair< keywords::ynodes, keywords::ynodes_value >,
                keyword_value_pair< keywords::znodes, keywords::znodes_value >,
                keyword_value_pair< keywords::xstepsize, keywords::xstepsize_value >,
                keyword_value_pair< keywords::ystepsize, keywords::ystepsize_value >,
                keyword_value_pair< keywords::zstepsize, keywords::zstepsize_value >,
                keyword_value_pair< keywords::xmin, keywords::xmin_value >,
                keyword_value_pair< keywords::ymin, keywords::ymin_value >,
                keyword_value_pair< keywords::zmin, keywords::zmin_value >,
                keyword_value_pair< keywords::xmax, keywords::xmax_value >,
                keyword_value_pair< keywords::ymax, keywords::ymax_value >,
                keyword_value_pair< keywords::zmax, keywords::zmax_value >,
                keyword_value_pair< keywords::xbase, keywords::xbase_value >,
                keyword_value_pair< keywords::ybase, keywords::ybase_value >,
                keyword_value_pair< keywords::zbase, keywords::zbase_value >
                >
            {};

        struct aovf_keyword_value_line
            : pegtl::sor< 
                ovf_keyword_value_line,
                // Atomistic extension
                keyword_value_pair< keywords::meshtype, keywords::meshtype_value_lattice >,
                keyword_value_pair< keywords::anodes, keywords::anodes_value >,
                keyword_value_pair< keywords::bnodes, keywords::bnodes_value >,
                keyword_value_pair< keywords::cnodes, keywords::cnodes_value >,
                keyword_value_pair< keywords::bravaisa, keywords::bravaisa_value >,
                keyword_value_pair< keywords::bravaisb, keywords::bravaisb_value >,
                keyword_value_pair< keywords::bravaisc, keywords::bravaisc_value >,
                keyword_value_pair< keywords::ncellpoints, keywords::ncellpoints_value >,
                keyword_value_pair< keywords::basis, keywords::basis_value >
             >
        {};

        struct caovf_keyword_value_line
            : pegtl::sor< 
                ovf_keyword_value_line,
                // Atomistic extension
                magic_keyword_value_pair< keywords::meshtype, keywords::meshtype_value_lattice >,
                magic_keyword_value_pair< keywords::anodes, keywords::anodes_value >,
                magic_keyword_value_pair< keywords::bnodes, keywords::bnodes_value >,
                magic_keyword_value_pair< keywords::cnodes, keywords::cnodes_value >,
                magic_keyword_value_pair< keywords::bravaisa, keywords::bravaisa_value >,
                magic_keyword_value_pair< keywords::bravaisb, keywords::bravaisb_value >,
                magic_keyword_value_pair< keywords::bravaisc, keywords::bravaisc_value >,
                magic_keyword_value_pair< keywords::ncellpoints, keywords::ncellpoints_value >,
                magic_keyword_value_pair< keywords::basis, keywords::basis_value >
             >
        {};

        template<typename keyword_value_line_t>
        struct header
            : pegtl::seq<
                begin, TAO_PEGTL_ISTRING("Header"), finish_line,
                pegtl::until<
                    pegtl::seq<end, TAO_PEGTL_ISTRING("Header")>,
                    pegtl::must<
                        skippable_lines,
                        keyword_value_line_t,
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
        template<typename keyword_value_line_t>
        struct segment_header
            : pegtl::seq<
                pegtl::star<pegtl::seq<empty_line, pegtl::eol>>,
                pegtl::seq< begin, TAO_PEGTL_ISTRING("Segment"), pegtl::eol>,
                pegtl::star<pegtl::seq<empty_line, pegtl::eol>>,
                header<keyword_value_line_t>,
                pegtl::star<pegtl::seq<empty_line, pegtl::eol>>,
                pegtl::until<pegtl::seq<end, TAO_PEGTL_ISTRING("Segment")>>, pegtl::eol >
        {};

        // Class template for user-defined actions that does nothing by default.
        template< typename Rule >
        struct ovf_segment_header_action : keywords::kw_action<Rule>
        {};

        template<typename keyword_value_line_t>
        struct ovf_segment_header_action< segment_header<keyword_value_line_t> >
        {
            template< typename Input >
            static void apply( const Input& in, ovf_file & file, ovf_segment & segment )
            {

                // Check if all required keywords were present
                std::vector<std::string> missing_keywords(0);
                std::vector<std::string> wrong_keywords(0);

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

                if( std::string(segment.meshtype) == "rectangular" || (file.ovf_extension_format == OVF_EXTENSION_FORMAT_AOVF_COMP && file._state->found_meshtype_atomistic) )
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
                } else {
                    if( file._state->found_xbase )
                        wrong_keywords.push_back("xbase");
                    if( file._state->found_ybase )
                        wrong_keywords.push_back("ybase");
                    if( file._state->found_zbase )
                        wrong_keywords.push_back("zbase");
                    if( file._state->found_xstepsize )
                        wrong_keywords.push_back("xstepsize");
                    if( file._state->found_ystepsize )
                        wrong_keywords.push_back("ystepsize");
                    if( file._state->found_zstepsize )
                        wrong_keywords.push_back("zstepsize");
                    if( file._state->found_xnodes )
                        wrong_keywords.push_back("xnodes");
                    if( file._state->found_ynodes )
                        wrong_keywords.push_back("ynodes");
                    if( file._state->found_znodes )
                        wrong_keywords.push_back("znodes");
                }

                if( std::string(segment.meshtype) == "irregular" )
                {
                    segment.N = segment.pointcount;
                    if( !file._state->found_pointcount )
                        missing_keywords.push_back("pointcount");
                } else {
                    if( file._state->found_pointcount )
                        wrong_keywords.push_back("pointcount");
                }

                if( std::string(segment.meshtype) == "lattice" || file._state->found_meshtype_atomistic )
                {
                    segment.N = segment.n_cells[0] * segment.n_cells[1] * segment.n_cells[2] * segment.ncellpoints;
                    if( !file._state->found_anodes )
                        missing_keywords.push_back("anodes");
                    if( !file._state->found_bnodes )
                        missing_keywords.push_back("bnodes");
                    if( !file._state->found_cnodes )
                        missing_keywords.push_back("cnodes");
                    if( !file._state->found_ncellpoints )
                        missing_keywords.push_back("ncellpoints");
                    if( !file._state->found_basis )
                        missing_keywords.push_back("basis");
                    if( !file._state->found_bravaisa )
                        missing_keywords.push_back("bravaisa");
                    if( !file._state->found_bravaisb )
                        missing_keywords.push_back("bravaisb");
                    if( !file._state->found_bravaisc )
                        missing_keywords.push_back("bravaisc");
                } else {
                    if( file._state->found_anodes )
                        wrong_keywords.push_back("anodes");
                    if( file._state->found_bnodes )
                        wrong_keywords.push_back("bnodes");
                    if( file._state->found_cnodes )
                        wrong_keywords.push_back("cnodes");
                    if( file._state->found_ncellpoints )
                        wrong_keywords.push_back("ncellpoints");
                    if( file._state->found_basis )
                        wrong_keywords.push_back("basis");
                    if( file._state->found_bravaisa )
                        wrong_keywords.push_back("bravaisa");
                    if( file._state->found_bravaisb )
                        wrong_keywords.push_back("bravaisb");
                    if( file._state->found_bravaisc )
                        wrong_keywords.push_back("bravaisc");
                }

                if( missing_keywords.size() > 0 )
                {
                    std::string message = fmt::format( "Missing keywords for meshtype \"{}\": \"{}\"", segment.meshtype, missing_keywords[0] );
                    for( int i=1; i < missing_keywords.size(); ++i )
                        message += fmt::format( ", \"{}\"", missing_keywords[i] );
                    throw tao::pegtl::parse_error( message, in );
                }

                if( wrong_keywords.size() > 0 )
                {
                    std::string message = fmt::format( "Wrong keywords for meshtype \"{}\": \"{}\"", segment.meshtype, wrong_keywords[0] );
                    for( int i=1; i < wrong_keywords.size(); ++i )
                        message += fmt::format( ", \"{}\"", wrong_keywords[i] );
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
                check_value_bin_4,
                bytes_bin_4,
                pegtl::seq<end, TAO_PEGTL_ISTRING("Data Binary 4"), pegtl::eol>
                >
        {};

        struct data_binary_8
            : pegtl::seq<
                begin, TAO_PEGTL_ISTRING("Data Binary 8"), pegtl::eol,
                check_value_bin_8,
                bytes_bin_8,
                pegtl::seq<end, TAO_PEGTL_ISTRING("Data Binary 8"), pegtl::eol>
                >
        {};

        template<typename keyword_value_line_t>
        struct segment_data
            : pegtl::seq<
                pegtl::star<pegtl::seq<empty_line, pegtl::eol>>,
                begin, TAO_PEGTL_ISTRING("Segment"), pegtl::eol,
                pegtl::star<pegtl::seq<empty_line, pegtl::eol>>,
                header<keyword_value_line_t>,
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
        struct ovf_segment_data_action< text_value_float >
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
            }
        };

        template<>
        struct ovf_segment_data_action< check_value_bin_4 >
        {
            template< typename Input, typename scalar >
            static void apply( const Input& in, ovf_file & f, const ovf_segment & segment, scalar * data )
            {
                std::string bytes = in.string();
                uint32_t hex_4b = endian::from_little_32(reinterpret_cast<const uint8_t *>( bytes.c_str() ));

                if( hex_4b != check::val_4b )
                    throw tao::pegtl::parse_error( "the expected binary check value could not be parsed!", in );
            }
        };

        template<>
        struct ovf_segment_data_action< bytes_bin_4 >
        {
            template< typename Input, typename scalar >
            static void apply( const Input& in, ovf_file & f, const ovf_segment & segment, scalar * data )
            {
                std::string bytes_str = in.string();
                const uint8_t * bytes = reinterpret_cast<const uint8_t *>( bytes_str.c_str() );
                for( int idx=0; idx < f._state->max_data_index; ++idx )
                {
                    uint32_t ivalue = endian::from_little_32( &bytes[4*idx] );
                    float value = *reinterpret_cast<const float *>( &ivalue );

                    if( idx < f._state->max_data_index )
                    {
                        data[idx] = value;
                        ++f._state->current_column;
                    }

                    if( f._state->current_column > segment.valuedim )
                    {
                        f._state->current_column = 0;
                        ++f._state->current_line;
                    }
                }
                f._state->current_line = 0;
                f._state->current_column = 0;
            }
        };

        template<>
        struct ovf_segment_data_action< check_value_bin_8 >
        {
            template< typename Input, typename scalar >
            static void apply( const Input& in, ovf_file & f, const ovf_segment & segment, scalar * data )
            {
                std::string bytes = in.string();
                uint64_t hex_8b = endian::from_little_64(reinterpret_cast<const uint8_t *>( bytes.c_str() ));

                if( hex_8b != check::val_8b )
                    throw tao::pegtl::parse_error( "the expected binary check value could not be parsed!", in );
            }
        };

        template<>
        struct ovf_segment_data_action< bytes_bin_8 >
        {
            template< typename Input, typename scalar >
            static void apply( const Input& in, ovf_file & f, const ovf_segment & segment, scalar * data )
            {
                std::string bytes_str = in.string();
                const uint8_t * bytes = reinterpret_cast<const uint8_t *>( bytes_str.c_str() );
                for( int idx=0; idx < f._state->max_data_index; ++idx )
                {
                    uint64_t ivalue = endian::from_little_64( &bytes[8*idx] );
                    double value = *reinterpret_cast<const double *>( &ivalue );

                    if( idx < f._state->max_data_index )
                    {
                        data[idx] = value;
                        ++f._state->current_column;
                    }

                    if( f._state->current_column > segment.valuedim )
                    {
                        f._state->current_column = 0;
                        ++f._state->current_line;
                    }
                }
                f._state->current_line = 0;
                f._state->current_column = 0;
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
        void ovf_segment_header_control< ovf_keyword_value_line >::raise( const Input& in, States&&... )
        {
            throw keyword_value_line_error( in );
        }

        template<> template< typename Input, typename... States >
        void ovf_segment_header_control< aovf_keyword_value_line >::raise( const Input& in, States&&... )
        {
            throw keyword_value_line_error( in );
        }

        template<> template< typename Input, typename... States >
        void ovf_segment_header_control< caovf_keyword_value_line >::raise( const Input& in, States&&... )
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