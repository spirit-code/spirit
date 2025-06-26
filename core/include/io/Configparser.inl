#pragma once
#ifndef SPIRIT_CORE_IO_CONFIGPARSER_INL
#define SPIRIT_CORE_IO_CONFIGPARSER_INL
#include <Spirit/Spirit_Defines.h>
#include <engine/Vectormath_Defines.hpp>
#include <utility/Exception.hpp>

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

// Primary template: fallback for scalar types
template<typename T, typename Enable = void>
struct toml_array_transform
{
    using value_type = T;

    [[nodiscard]] static auto transform( const toml::node & node ) -> T
    {
        if( auto v = node.as<T>() )
            return v->get();
        else
            spirit_throw(
                Utility::Exception_Classifier::Input_parse_failed, Utility::Log_Level::Error,
                "Error while parsing toml array: invalid node type" );
    }
};

// Partial specialization for std::array<T,N>
template<typename T, std::size_t N>
struct toml_array_transform<std::array<T, N>>
{
    using container_type = std::array<T, N>;

    [[nodiscard]] static auto transform( const toml::node & node ) -> container_type
    {
        if( auto array = node.as_array(); array && array->size() == N )
        {
            std::array<T, N> result{};
            std::transform( array->begin(), array->end(), result.begin(), toml_array_transform<T>::transform );
            return result;
        }
        spirit_throw(
            Utility::Exception_Classifier::Input_parse_failed, Utility::Log_Level::Error,
            fmt::format( "Invalid array size while parsing toml array, expected {}", N ) );
    }
};

// Partial specialization for Vector3
template<>
struct toml_array_transform<Vector3>
{
    using container_type = Vector3;

    [[nodiscard]] static container_type transform( const toml::node & node )
    {
        if( auto array = node.as_array(); array && array->size() == 3 )
        {
            Vector3 result{};
            std::transform( array->begin(), array->end(), result.begin(), toml_array_transform<scalar>::transform );
            return result;
        }
        spirit_throw(
            Utility::Exception_Classifier::Input_parse_failed, Utility::Log_Level::Error,
            fmt::format( "Invalid array size while parsing toml array, expected {}", 3 ) );
    }
};

// Partial specialization for variable-length containers (e.g., std::vector, std::deque, etc.)
template<template<typename, typename...> class Container, typename T, typename... Args>
struct toml_array_transform<Container<T, Args...>>
{
    using container_type = Container<T, Args...>;

    [[nodiscard]] static auto transform( const toml::node & node ) -> container_type
    {
        if( auto array = node.as_array() )
        {
            container_type result{};
            result.reserve( array->size() );
            std::transform(
                array->begin(), array->end(), std::back_inserter( result ), toml_array_transform<T>::transform );
            return result;
        }
        spirit_throw(
            Utility::Exception_Classifier::Input_parse_failed, Utility::Log_Level::Error,
            "Error while parsing toml array: not an array type" );
    }
};

template<>
struct toml_array_transform<toml::array>
{
    using container_type = toml::array;

    template<typename Container>
    [[nodiscard]] static auto transform( const Container & container ) noexcept -> toml::array
    {
        using T = typename Container::value_type;
        toml::array array;
        if constexpr( std::is_floating_point_v<T> || std::is_integral_v<T> )
            array.insert( array.begin(), container.begin(), container.end() );
        else
            std::transform(
                container.begin(), container.end(), std::back_inserter( array ),
                toml_array_transform<toml::array>::transform<T> );
        return array;
    };
};

template<typename Container>
auto toml_array_from_container( const Container & iterable ) -> toml::array
{
    return toml_array_transform<toml::array>::transform( iterable );
    // toml::array result{};
    // result.insert( result.begin(), iterable.begin(), iterable.end() );
    // return result;
}

} // namespace IO
#endif
