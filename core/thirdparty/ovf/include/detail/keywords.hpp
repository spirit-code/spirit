#pragma once
#ifndef LIBOVF_DETAIL_KEYWORDS_H
#define LIBOVF_DETAIL_KEYWORDS_H
#include "ovf.h"
#include "pegtl_defines.hpp"

#include <string>
#include <detail/helpers.hpp>

#include <tao/pegtl.hpp>
#include <fmt/format.h>

#include <array>


namespace ovf 
{
namespace detail
{

namespace keywords
{
    using ovf::detail::parse::decimal_number;
    namespace pegtl = tao::pegtl;

    template< typename Rule >
    struct kw_action
        : pegtl::nothing< Rule >
    { };

    struct end_kw_value : pegtl::at< pegtl::sor<pegtl::eol, pegtl::istring<'#'>> > {};
    struct standard_kw_value : pegtl::until< end_kw_value > {};
    struct numeric_kw_value : pegtl::seq<pegtl::pad< decimal_number, pegtl::blank >, end_kw_value > {};

    ////// title
    struct title : TAO_PEGTL_ISTRING("title")
    { };

    struct title_value : standard_kw_value
    { };

    template<>
    struct kw_action< title_value >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            segment.title = strdup(in.string().c_str());
            f._state->found_title = true;
        }
    };

    ////// desc
    struct desc : TAO_PEGTL_ISTRING("desc")
    { };

    struct desc_value : standard_kw_value
    { };

    ////// valuedim
    struct valuedim : TAO_PEGTL_ISTRING("valuedim")
    { };

    struct valuedim_value : standard_kw_value
    { };

    template<>
    struct kw_action< valuedim_value >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            segment.valuedim = std::stoi(in.string());
            f._state->found_valuedim = true;
        }
    };

    ////// valueunits
    struct valueunits : TAO_PEGTL_ISTRING("valueunits")
    { };

    struct valueunits_value : standard_kw_value
    { };

    template<>
    struct kw_action< valueunits_value >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            segment.valueunits = strdup(in.string().c_str());
            f._state->found_valueunits = true;
        }
    };

    ////// valuelabels
    struct valuelabels : TAO_PEGTL_ISTRING("valuelabels")
    { };

    struct valuelabels_value : standard_kw_value
    { };

    template<>
    struct kw_action< valuelabels_value >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            segment.valuelabels = strdup(in.string().c_str());
            f._state->found_valuelabels = true;
        }
    };

    ////// meshtype
    struct meshtype : TAO_PEGTL_ISTRING("meshtype")
    { };

    struct meshtype_value : pegtl::sor< TAO_PEGTL_ISTRING("rectangular"), TAO_PEGTL_ISTRING("irregular")> // Only 'rectangular' and 'irregular' allowed
    { };

    template<>
    struct kw_action< meshtype_value >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            segment.meshtype = strdup(in.string().c_str());
            f._state->found_meshtype = true;
        }
    };

    ////// meshunit
    struct meshunit : TAO_PEGTL_ISTRING("meshunit")
    { };

    struct meshunit_value : standard_kw_value
    { };

    template<>
    struct kw_action< meshunit_value >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            segment.meshunit = strdup(in.string().c_str());
            f._state->found_meshunit = true;
        }
    };

    ////// pointcount
    struct pointcount : TAO_PEGTL_ISTRING("pointcount")
    { };

    struct pointcount_value : numeric_kw_value
    { };

    template<>
    struct kw_action< pointcount_value >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            segment.pointcount = std::stoi(in.string());
            f._state->found_pointcount = true;
        }
    };

    ////// xnodes
    struct xnodes : TAO_PEGTL_ISTRING("xnodes")
    { };

    struct xnodes_value : numeric_kw_value
    { };

    template<>
    struct kw_action< xnodes_value >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            segment.n_cells[0] = std::stoi(in.string());
            f._state->found_xnodes = true;
        }
    };

    ////// ynodes
    struct ynodes : TAO_PEGTL_ISTRING("ynodes")
    { };

    struct ynodes_value : numeric_kw_value
    { };

    template<>
    struct kw_action< ynodes_value >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            segment.n_cells[1] = std::stoi(in.string());
            f._state->found_ynodes = true;
        }
    };

    ////// znodes
    struct znodes : TAO_PEGTL_ISTRING("znodes")
    { };

    struct znodes_value : numeric_kw_value
    { };

    template<>
    struct kw_action< znodes_value >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            segment.n_cells[2] = std::stoi(in.string());
            f._state->found_znodes = true;
        }
    };

    ////// xstepsize
    struct xstepsize : TAO_PEGTL_ISTRING("xstepsize")
    { };

    struct xstepsize_value : numeric_kw_value
    { };

    template<>
    struct kw_action< xstepsize_value >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            segment.step_size[0] = std::stof(in.string());
            f._state->found_xstepsize = true;
        }
    };

    ////// ystepsize
    struct ystepsize : TAO_PEGTL_ISTRING("ystepsize")
    { };

    struct ystepsize_value : numeric_kw_value
    { };

    template<>
    struct kw_action< ystepsize_value >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            segment.step_size[1] = std::stof(in.string());
            f._state->found_ystepsize = true;
        }
    };

    ////// zstepsize
    struct zstepsize : TAO_PEGTL_ISTRING("zstepsize")
    { };

    struct zstepsize_value : numeric_kw_value
    { };

    template<>
    struct kw_action< zstepsize_value >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            segment.step_size[2] = std::stof(in.string());
            f._state->found_zstepsize = true;
        }
    };

    ////// xmin
    struct xmin : TAO_PEGTL_ISTRING("xmin")
    { };

    struct xmin_value : numeric_kw_value
    { };

    template<>
    struct kw_action< xmin_value >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            segment.bounds_min[0] = std::stof(in.string());
            f._state->found_xmin = true;
        }
    };

    ////// ymin
    struct ymin : TAO_PEGTL_ISTRING("ymin")
    { };

    struct ymin_value : numeric_kw_value
    { };

    template<>
    struct kw_action< ymin_value >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            segment.bounds_min[1] = std::stof(in.string());
            f._state->found_ymin = true;
        }
    };

    ////// zmin
    struct zmin : TAO_PEGTL_ISTRING("zmin")
    { };

    struct zmin_value : numeric_kw_value
    { };

    template<>
    struct kw_action< zmin_value >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            segment.bounds_min[2] = std::stof(in.string());
            f._state->found_zmin = true;
        }
    };

    ////// xmax
    struct xmax : TAO_PEGTL_ISTRING("xmax")
    { };

    struct xmax_value : numeric_kw_value
    { };

    template<>
    struct kw_action< xmax_value >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            segment.bounds_max[0] = std::stof(in.string());
            f._state->found_xmax = true;
        }
    };

    ////// ymax
    struct ymax : TAO_PEGTL_ISTRING("ymax")
    { };

    struct ymax_value : numeric_kw_value
    { };

    template<>
    struct kw_action< ymax_value >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            segment.bounds_max[1] = std::stof(in.string());
            f._state->found_ymax = true;
        }
    };

    ////// zmax
    struct zmax : TAO_PEGTL_ISTRING("zmax")
    { };

    struct zmax_value : numeric_kw_value
    { };

    template<>
    struct kw_action< zmax_value >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            segment.bounds_max[2] = std::stof(in.string());
            f._state->found_zmax = true;
        }
    };

    ////// xbase
    struct xbase : TAO_PEGTL_ISTRING("xbase")
    { };

    struct xbase_value : numeric_kw_value
    { };

    template<>
    struct kw_action< xbase_value >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            segment.origin[0] = std::stof(in.string());
            f._state->found_xbase = true;
        }
    };

    ////// ybase
    struct ybase : TAO_PEGTL_ISTRING("ybase")
    { };

    struct ybase_value : numeric_kw_value
    { };

    template<>
    struct kw_action< ybase_value >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            segment.origin[1] = std::stof(in.string());
            f._state->found_ybase = true;
        }
    };

    ////// zbase
    struct zbase : TAO_PEGTL_ISTRING("zbase")
    { };

    struct zbase_value : numeric_kw_value
    { };

    template<>
    struct kw_action< zbase_value >
    {
        template< typename Input >
        static void apply( const Input& in, ovf_file & f, ovf_segment & segment)
        {
            segment.origin[2] = std::stof(in.string());
            f._state->found_zbase = true;
        }
    };

}
}
}

#endif