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

    // set up some utilities for parsing vector3
    namespace Vector3 = ovf::detail::parse::Vector3;

    using vec3_t = float[3];

    template<typename Rule>
    using vec3_action_t = Vector3::action<Rule, vec3_t>;

    template<class Input>
    inline void read_vector(const Input & in, vec3_t & vec3_data)
    {
        Vector3::read_vec3<const Input &, vec3_t, vec3_action_t>(in, vec3_data);
    }

    ////// meshtype

    // ONLY TRIGGERS ON MESHTYPE LATTICE, the rest of the meshtype keyword is implemented in keywords.hpp
    struct meshtype_value_lattice : TAO_PEGTL_ISTRING("lattice") // This keyword is triggered only, when meshtype lattice is found
    { };

    template<>
    struct kw_action< meshtype_value_lattice >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            segment.meshtype = strdup(in.string().c_str());
            f._state->found_meshtype = true;
            f._state->found_meshtype_lattice = true;
        }
    };

    ////// bravaisa
    struct bravaisa : TAO_PEGTL_ISTRING("bravaisa")
    { };

    struct bravaisa_value : pegtl::seq<Vector3::vec3, end_kw_value> {};
    template<>
    struct kw_action< bravaisa_value >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            f._state->found_bravaisa = true;
            read_vector(in, segment.bravaisa);
        }
    };

    ////// bravaisb
    struct bravaisb : TAO_PEGTL_ISTRING("bravaisb")
    { };

    struct bravaisb_value : pegtl::seq<Vector3::vec3, end_kw_value> {};
    template<>
    struct kw_action< bravaisb_value >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            f._state->found_bravaisb = true;
            read_vector(in, segment.bravaisb);
        }
    };

    ////// bravaisc
    struct bravaisc : TAO_PEGTL_ISTRING("bravaisc")
    { };

    struct bravaisc_value : pegtl::seq<Vector3::vec3, end_kw_value> {};
    template<>
    struct kw_action< bravaisc_value >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            f._state->found_bravaisc = true;
            read_vector(in, segment.bravaisc);
        }
    };

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
            f._state->_basis.reserve(segment.ncellpoints); // If we find ncellpoints, we reserve the space
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

    struct basis_value_line : Vector3::vec3
    { };

    struct basis_value : pegtl::seq< pegtl::eol, pegtl::list<pegtl::seq< pegtl::string<'#'>, pegtl::opt<TAO_PEGTL_ISTRING("#%")>, basis_value_line>, pegtl::eol > > 
    { };

    template<>
    struct kw_action< basis_value_line >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            float temp[3];
            read_vector( in, temp );
            f._state->_basis.push_back( {temp[0], temp[1], temp[2]} );
        }
    };

    template<>
    struct kw_action<basis_value>
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            f._state->found_basis = true;
            // Allocate and data in segment struct and copy
            segment.basis = new float[3 * f._state->_basis.size()];
            for( int i=0; i<f._state->_basis.size(); i++)
            {
                segment.basis[3*i] = f._state->_basis[i][0];
                segment.basis[3*i + 1] = f._state->_basis[i][1];
                segment.basis[3*i + 2] = f._state->_basis[i][2];
            }
        }
    };
}
}
}

#endif