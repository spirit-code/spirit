#pragma once
#ifndef SPIRIT_CORE_IO_CONFIGPARSER_INL
#define SPIRIT_CORE_IO_CONFIGPARSER_INL
#include <Spirit/Spirit_Defines.h>
#include <engine/Vectormath_Defines.hpp>
#include <io/Configparser.hpp>
#include <utility/Exception.hpp>
#include <utility/Type_Traits.hpp>

#include <toml++/toml.hpp>

namespace IO
{

template<typename T>
void read_value( const toml::table & tbl, std::string_view key, T & dest, bool log_missing = true ) noexcept
{
    using namespace Utility;
    const auto node = tbl.at_path( key );
    if( !node )
    {
        if( log_missing )
            Log( Log_Level::Warning, Log_Sender::IO,
                 fmt::format( "Missing key encountererd!: '{}', using default: {}", key, dest ) );
        return;
    }

    if( auto result = toml_transform<T>( *node.node() ) )
        dest = *result;
    else
        Log( Log_Level::Error, Log_Sender::IO,
             fmt::format( "Failed converting value for key: '{}', using default: {}", key, dest ) );
}
namespace detail
{

// Primary template: fallback for scalar types
template<typename T, typename Enable = void>
struct toml_array_transform
{
    using value_type = T;

    [[nodiscard]] static auto transform( const toml::node & node ) -> T
    {
        if constexpr( std::is_integral_v<T> && !std::is_same_v<T, bool> )
        {
            if( auto v = node.value<int64_t>() )
                return static_cast<T>( *v );
        }
        else if constexpr( std::is_floating_point_v<T> )
        {
            if( auto v = node.value<double>() )
                return static_cast<T>( *v );
            if( auto v = node.value<int64_t>() )
                return static_cast<T>( *v );
        }
        else if constexpr( std::is_same_v<T, bool> || std::is_same_v<T, std::string> )
        {
            if( auto v = node.value<T>() )
                return *v;
        }
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
struct toml_array_transform<Container<T, Args...>, std::enable_if_t<!Utility::is_string_v<Container<T, Args...>>>>
{
    using container_type = Container<T, Args...>;

    [[nodiscard]] static auto transform( const toml::node & node ) -> container_type
    {
        if( auto array = node.as_array() )
        {
            container_type result{};
            result.reserve( array->size() );
            for( const auto & element : *array )
            {
                result.emplace_back( toml_array_transform<T>::transform( element ) );
            }
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
        array.reserve( container.size() );
        if constexpr(
            std::is_floating_point_v<T> || std::is_integral_v<T> || std::is_same_v<T, std::string>
            || std::is_same_v<T, std::string_view> )
        {
            array.insert( array.begin(), container.begin(), container.end() );
        }
        else
        {
            for( const auto & element : container )
            {
                array.emplace_back( toml_array_transform<toml::array>::transform( element ) );
            }
        }
        return array;
    };
};

} // namespace detail

template<typename T>
auto toml_transform( const toml::node & node ) noexcept -> std::optional<T>
{
    std::optional<T> result;
    try
    {
        result.emplace( detail::toml_array_transform<T>::transform( node ) );
    }
    catch( ... )
    {
        spirit_handle_exception_core( "Error while converting toml node." );
    }
    return result;
}

template<typename T>
auto toml_transform( toml::node_view<const toml::node> node_view ) noexcept -> std::optional<T>
{
    if( node_view )
        return toml_transform<T>( *node_view.node() );
    else
        return std::nullopt;
}

template<typename Container>
auto toml_array_from_container( const Container & iterable ) -> toml::array
{
    return detail::toml_array_transform<toml::array>::transform( iterable );
}

} // namespace IO
#endif
