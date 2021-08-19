#pragma once
#ifndef LIBOVF_DETAIL_PEGTL_DEFINES_H
#define LIBOVF_DETAIL_PEGTL_DEFINES_H

#include <array>
#include <vector>
#include <string>
#include <ios>
#include <tao/pegtl.hpp>

struct parser_state
{
    // For the segment strings
    std::vector<std::string> file_contents{};

    // for reading data blocks
    int current_column = 0;
    int current_line = 0;

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
    bool found_bravaisa     = false;
    bool found_bravaisb     = false;
    bool found_bravaisc     = false;
    bool found_ncellpoints  = false;
    bool found_anodes       = false;
    bool found_bnodes       = false;
    bool found_cnodes       = false;
    bool found_basis        = false;

    // Needed to keep track of the current line when reading in the basis positions
    bool found_meshtype_atomistic = false;
    int _cur_basis_line = 0;

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

}
}
}

#endif