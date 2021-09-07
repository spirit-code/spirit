#pragma once
#ifndef LIBOVF_DETAIL_PEGTL_DEFINES_H
#define LIBOVF_DETAIL_PEGTL_DEFINES_H

#include <array>
#include <vector>
#include <string>
#include <ios>
#include <tao/pegtl.hpp>
#include <iostream>

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

    /*
    We need and additional bool, because in the compatibiliby format, we can have:
            # meshtype : rectangular
            ##% meshtype : lattice
    So if the meshtype is rectangula, but found_meshtype_lattice is true we know the lattice meshtype was requested in the CAOVF format
    */
    bool found_meshtype_lattice = false;

    std::vector<std::array<float, 3>> _basis = std::vector<std::array<float, 3>>(0);
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

    namespace Vector3
    {
        struct x_val : tao::pegtl::pad<decimal_number, pegtl::blank> {};
        struct y_val : tao::pegtl::pad<decimal_number, pegtl::blank> {};
        struct z_val : tao::pegtl::pad<decimal_number, pegtl::blank> {};
        struct vec3 : tao::pegtl::seq<x_val, y_val, z_val> {};

        template<typename Rule, typename vec3_t>
        struct action
            : pegtl::nothing< Rule >
        { };

        template<typename vec3_t>
        struct action< x_val, vec3_t>
        {
            template< typename Input >
            static void apply( const Input& in, vec3_t & data)
            {
                data[0] = std::stof(in.string());
            }
        };

        template<typename vec3_t>
        struct action< y_val, vec3_t >
        {
            template< typename Input >
            static void apply( const Input& in, vec3_t & data )
            {
                data[1] = std::stof(in.string());
            }
        };

        template<typename vec3_t>
        struct action<z_val, vec3_t>
        {
            template< typename Input >
            static void apply( const Input& in, vec3_t & data)
            {
                data[2] = std::stof(in.string());
            }
        };

        template<typename input_t, typename vec3_t, template<class> class action_t>
        inline void read_vec3(input_t & in, vec3_t & vec_data)
        {
            std::string in_str = std::string(in.string());
            pegtl::memory_input<pegtl::tracking_mode::lazy, pegtl::eol::lf_crlf, std::string > in_mem( in_str, "" );
            bool success = pegtl::parse<pegtl::seq<vec3, pegtl::star<pegtl::any>>, action_t>(in_mem, vec_data);
        }

    }

}
}
}

#endif