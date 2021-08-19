#pragma once
#ifndef LIBOVF_DETAIL_ATOMISTIC_KEYWORDS_H
#define LIBOVF_DETAIL_ATOMISTIC_KEYWORDS_H
#include "keywords.hpp"

namespace ovf 
{
namespace detail
{
namespace keywords
{

    ////// meshtype

    // Only reimplement the new value
    struct meshtype_value_lattice : TAO_PEGTL_ISTRING("lattice") // Only 'rectangular', 'irregular' or 'lattice' allowed
    { };

    template<>
    struct kw_action< meshtype_value_lattice >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            segment.meshtype = strdup(in.string().c_str());
            f._state->found_meshtype_atomistic = true;
            f._state->found_meshtype = true;
        }
    };

    ////// bravaisa
    // x
    struct bravaisa_value_x : pegtl::pad<ovf::detail::parse::decimal_number, pegtl::blank>
    { };

    template<>
    struct kw_action< bravaisa_value_x >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            segment.bravaisa[0] = std::stof(in.string());
        }
    };

    // y
    struct bravaisa_value_y : pegtl::pad<decimal_number, pegtl::blank>
    { };

    template<>
    struct kw_action< bravaisa_value_y >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            segment.bravaisa[1] = std::stof(in.string());
        }
    };

    // z
    struct bravaisa_value_z : pegtl::pad<decimal_number, pegtl::blank>
    { };

    template<>
    struct kw_action< bravaisa_value_z >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            segment.bravaisa[2] = std::stof(in.string());
        }
    };

    struct bravaisa_value : pegtl::seq<bravaisa_value_x, bravaisa_value_y, bravaisa_value_z, end_kw_value>
    { };

    template<>
    struct kw_action< bravaisa_value >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            f._state->found_bravaisa = true;
        }
    };

    struct bravaisa : TAO_PEGTL_ISTRING("bravaisa")
    { };

    ////// bravaisb
    // x
    struct bravaisb_value_x : pegtl::pad<ovf::detail::parse::decimal_number, pegtl::blank>
    { };

    template<>
    struct kw_action< bravaisb_value_x >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            segment.bravaisb[0] = std::stof(in.string());
        }
    };

    // y
    struct bravaisb_value_y : pegtl::pad<decimal_number, pegtl::blank>
    { };

    template<>
    struct kw_action< bravaisb_value_y >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            segment.bravaisb[1] = std::stof(in.string());
        }
    };

    // z
    struct bravaisb_value_z : pegtl::pad<decimal_number, pegtl::blank>
    { };

    template<>
    struct kw_action< bravaisb_value_z >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            segment.bravaisb[2] = std::stof(in.string());
        }
    };

    struct bravaisb_value : pegtl::seq<bravaisb_value_x, bravaisb_value_y, bravaisb_value_z, end_kw_value>
    { };

    template<>
    struct kw_action< bravaisb_value >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            f._state->found_bravaisb = true;
        }
    };

    struct bravaisb : TAO_PEGTL_ISTRING("bravaisb")
    { };

    ////// bravaisc
    // x
    struct bravaisc_value_x : pegtl::pad<ovf::detail::parse::decimal_number, pegtl::blank>
    { };

    template<>
    struct kw_action< bravaisc_value_x >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            segment.bravaisc[0] = std::stof(in.string());
        }
    };

    // y
    struct bravaisc_value_y : pegtl::pad<decimal_number, pegtl::blank>
    { };

    template<>
    struct kw_action< bravaisc_value_y >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            segment.bravaisc[1] = std::stof(in.string());
        }
    };

    // z
    struct bravaisc_value_z : pegtl::pad<decimal_number, pegtl::blank>
    { };

    template<>
    struct kw_action< bravaisc_value_z >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            segment.bravaisc[2] = std::stof(in.string());
        }
    };

    struct bravaisc_value : pegtl::seq<bravaisc_value_x, bravaisc_value_y, bravaisc_value_z, end_kw_value>
    { };

    template<>
    struct kw_action< bravaisc_value >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            f._state->found_bravaisc = true;
        }
    };

    struct bravaisc : TAO_PEGTL_ISTRING("bravaisc")
    { };


    ////// ncellpoints
    struct ncellpoints : TAO_PEGTL_ISTRING("ncellpoints")
    { };

    struct ncellpoints_value : numeric_kw_value
    { };

    template<>
    struct kw_action< ncellpoints_value >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            segment.ncellpoints = std::stoi(in.string());
            f._state->found_ncellpoints = true;
            segment.basis = new float[3 * segment.ncellpoints];
        }
    };

    ////// anodes
    struct anodes : TAO_PEGTL_ISTRING("anodes")
    { };

    struct anodes_value : numeric_kw_value
    { };

    template<>
    struct kw_action< anodes_value >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            segment.n_cells[0] = std::stoi(in.string());
            f._state->found_anodes = true;
        }
    };

    ////// bnodes
    struct bnodes : TAO_PEGTL_ISTRING("bnodes")
    { };

    struct bnodes_value : numeric_kw_value
    { };

    template<>
    struct kw_action< bnodes_value >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            segment.n_cells[1] = std::stoi(in.string());
            f._state->found_bnodes = true;
        }
    };

    ////// cnodes
    struct cnodes : TAO_PEGTL_ISTRING("cnodes")
    { };

    struct cnodes_value : numeric_kw_value
    { };

    template<>
    struct kw_action< cnodes_value >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            segment.n_cells[2] = std::stoi(in.string());
            f._state->found_cnodes = true;
        }
    };

    ////// basis
    struct basis : TAO_PEGTL_ISTRING("basis")
    { };

    struct cur_basis_line_value_x : pegtl::pad<decimal_number, pegtl::blank>
    { };

    struct cur_basis_line_value_y : pegtl::pad<decimal_number, pegtl::blank>
    { };

    struct cur_basis_line_value_z : pegtl::pad<decimal_number, pegtl::blank>
    { };

    struct basis_value_line : pegtl::seq< pegtl::string<'#'>, pegtl::opt<TAO_PEGTL_ISTRING("#%")>, cur_basis_line_value_x, cur_basis_line_value_y, cur_basis_line_value_z>
    { };

    struct basis_value : pegtl::seq< pegtl::eol, pegtl::list<basis_value_line, pegtl::eol > > 
    { };

    template<>
    struct kw_action< cur_basis_line_value_x >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            if( !f._state->found_ncellpoints ) // Need to make sure that the basis array is already allocated
            {
                throw tao::pegtl::parse_error(fmt::format("ncellpoints must be specified before the basis!"), in);
            }
            segment.basis[3 * f._state->_cur_basis_line + 0] = std::stof(in.string());
        }
    };

    template<>
    struct kw_action< cur_basis_line_value_y >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            if( !f._state->found_ncellpoints ) // Need to make sure that the basis array is already allocated
            {
                throw tao::pegtl::parse_error(fmt::format("ncellpoints must be specified before the basis!"), in);
            }
            segment.basis[3*f._state->_cur_basis_line + 1] = std::stof(in.string());
        }
    };

    template<>
    struct kw_action< cur_basis_line_value_z >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            if( !f._state->found_ncellpoints ) // Need to make sure that the basis array is already allocated
            {
                throw tao::pegtl::parse_error(fmt::format("ncellpoints must be specified before the basis!"), in);
            }
            segment.basis[3* f._state->_cur_basis_line + 2] = std::stof(in.string());
        }
    };

    template<>
    struct kw_action< basis_value_line >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            // fmt::print("basis line value: {}\n", in.string());
            f._state->_cur_basis_line++;
        }
    };

    template<>
    struct kw_action<basis_value>
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            // fmt::print("basis value: {}\n", in.string());

            f._state->found_basis = true;
            if( segment.ncellpoints != f._state->_cur_basis_line ) // Need to make sure that the basis array is already allocated
            {
                throw tao::pegtl::parse_error( fmt::format("ncellpoints ({}) and number of specified basis atoms ({}) does not match!", segment.ncellpoints, f._state->_cur_basis_line ), in);
            }
        }
    };
}
}
}

#endif