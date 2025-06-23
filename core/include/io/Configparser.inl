#pragma once
#ifndef SPIRIT_CORE_IO_CONFIGPARSER_INL
#define SPIRIT_CORE_IO_CONFIGPARSER_INL
#include <Spirit/Spirit_Defines.h>
#include <engine/Vectormath_Defines.hpp>

#include <toml++/toml.hpp>

namespace IO
{

template<typename T>
void read_single( T && dest, toml::node_view<const toml::node> node ) noexcept
{
    dest = node.value_or( dest );
}

template<typename T, typename U>
void read_single( T && dest, toml::node_view<const toml::node> node, U && default_value ) noexcept
{
    dest = node.value_or( std::forward<U>( default_value ) );
}

inline void read_Vector3( Vector3 & dest, toml::node_view<const toml::node> node ) noexcept
{
    if( auto arr = node.as_array(); arr && arr->size() == 3 )
    {
        Vector3 result{};
        for( int i = 0; i < 3; ++i )
        {
            if( auto value = ( *arr )[i].value<scalar>() )
                result[i] = *value;
            else
                return;
        }

        dest = result;
    }
}

} // namespace IO
#endif
