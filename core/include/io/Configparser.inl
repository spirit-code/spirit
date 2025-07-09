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

namespace detail
{

template<typename T>
auto pre_format( T && value ) -> decltype( auto )
{
    if constexpr( is_eigen_type_v<std::decay_t<T>> )
        return value.transpose();
    else if constexpr( std::is_enum_v<std::decay_t<T>> )
        return name( value );
    else
        return std::forward<T>( value );
}

} // namespace detail

template<typename T, typename = std::enable_if_t<!std::is_enum_v<T> && "Use `read_enum()` for enum types!">>
void read_value( const toml::table & tbl, std::string_view key, T & dest, bool log_missing = true ) noexcept
{
    static_assert( !std::is_enum_v<T> );

    using namespace Utility;
    const auto node = tbl.at_path( key );
    if( !node )
    {
        if( log_missing )
            Log( Log_Level::Warning, Log_Sender::IO,
                 fmt::format( "Missing key encountererd: '{}', using default: {}", key, detail::pre_format( dest ) ) );
        return;
    }

    if( auto result = toml_transform<T>( *node.node() ) )
        dest = *result;
    else
        Log( Log_Level::Error, Log_Sender::IO,
             fmt::format(
                 "Failed converting value for key: '{}', using default: {}", key, detail::pre_format( dest ) ) );
}

template<typename Enum, typename = std::enable_if_t<std::is_enum_v<Enum> && "Use `read_value() for non-enum types!`">>
void read_enum( const toml::table & tbl, std::string_view key, Enum & dest, bool log_missing = true ) noexcept
{
    using Utility::Enum::from_string;

    static_assert( std::is_enum_v<Enum> );
    static_assert( std::is_same_v<decltype( name( std::declval<Enum>() ) ), std::string_view> );
    static_assert( std::is_same_v<decltype( from_string<Enum>( std::declval<std::string>() ) ), std::optional<Enum>> );

    using namespace Utility;
    const auto node = tbl.at_path( key );
    if( !node )
    {
        if( log_missing )
            Log( Log_Level::Warning, Log_Sender::IO,
                 fmt::format( "Missing key encountererd: '{}', using default: {}", key, detail::pre_format( dest ) ) );
        return;
    }
    if( auto result_str = toml_transform<std::string>( *node.node() ) )
    {
        if( auto result_enum = from_string<Enum>( *result_str ) )
            dest = *result_enum;
        else
            Log( Log_Level::Error, Log_Sender::IO,
                 fmt::format(
                     "Unknown value '{}' for key: '{}', using default: {}", *result_str, key,
                     detail::pre_format( dest ) ) );
    }
    else
        Log( Log_Level::Error, Log_Sender::IO,
             fmt::format(
                 "Failed converting value for key: '{}', using default: {}", key, detail::pre_format( dest ) ) );
}

inline void read_Vector3(
    const toml::table & tbl, std::string_view key, scalar & dest_magnitude, Vector3 & dest_direction,
    bool log_missing = true ) noexcept
{
    const auto default_msg = [&dest_magnitude, &dest_direction]
    { return fmt::format( "using default: {{ magnitude = {}, direction = ({}) }}", dest_magnitude, dest_direction ); };

    using namespace Utility;
    const auto node = tbl.at_path( key );
    if( !node )
    {
        if( log_missing )
            Log( Log_Level::Warning, Log_Sender::IO,
                 fmt::format( "Missing key encountererd!: '{}', {}", key, default_msg() ) );
        return;
    }

    if( node.is_array() )
    {
        if( auto vec = toml_transform<Vector3>( node ) )
        {
            dest_magnitude = vec->norm();
            vec->normalize();
            if( vec->norm() > 1e-2 ) // This should be either 1 or close to zero.
                dest_direction = *vec;
            else
            {
                dest_magnitude = 0.0;
                Log( Log_Level::Warning, Log_Sender::IO,
                     fmt::format(
                         "Cannot deduce direction for key '{}', the provided vector is too close to zero. "
                         "Setting to {{ magnitue = {}, direction = ({}) }}",
                         key, dest_magnitude, dest_direction.transpose() ) );
            }
        }
        else
            Log( Log_Level::Error, Log_Sender::IO,
                 fmt::format( "Failed converting array to Vector3 for key: '{}', {}", key, default_msg() ) );
    }
    else if( const auto * node_tbl = node.as_table() )
    {
        static constexpr std::string_view m_key = "magnitude";
        static constexpr std::string_view n_key = "direction";

        bool found_magitude  = false;
        bool found_direction = false;
        for( const auto & [provided_key, provided_value] : *node_tbl )
        {
            if( provided_key == m_key )
            {
                found_magitude = true;
                if( auto magnitude = toml_transform<scalar>( provided_value ) )
                    dest_magnitude = *magnitude;
                else
                    Log( Log_Level::Warning, Log_Sender::IO,
                         fmt::format(
                             "Wrong type for key: '{}.{}', expected floating point, using defaukt {}", key, m_key,
                             dest_magnitude ) );
            }
            else if( provided_key == n_key )
            {
                found_direction = true;
                if( auto direction = toml_transform<Vector3>( provided_value ) )
                {
                    direction->normalize();
                    if( direction->norm() > 1e-2 )
                        dest_direction = *direction;
                    else
                        Log( Log_Level::Warning, Log_Sender::IO,
                             fmt::format(
                                 "Failed normalizing Vector3: '{}.{}', using defaukt {}", key, n_key,
                                 dest_direction.transpose() ) );
                }
                else
                    Log( Log_Level::Warning, Log_Sender::IO,
                         fmt::format(
                             "Failed converting node to Vector3: '{}.{}', using defaukt {}", key, n_key,
                             dest_direction.transpose() ) );
            }
            else
                Log( Log_Level::Warning, Log_Sender::IO,
                     fmt::format(
                         "Unknown key '{}' encountered in table: {}, valid keys are '{}' and '{}'", provided_key.str(),
                         key, m_key, n_key ) );
        }

        if( !found_magitude )
            Log( Log_Level::Warning, Log_Sender::IO,
                 fmt::format( "Missing key encountererd: '{}.{}', using default {}", key, m_key, dest_magnitude ) );
        if( !found_direction )
            Log( Log_Level::Warning, Log_Sender::IO,
                 fmt::format(
                     "Missing key encountererd!: '{}.{}', using default {}", key, n_key, dest_direction.transpose() ) );
    };
}

template<typename T>
void read_value_with_default(
    const toml::table & tbl, std::string_view key, T & dest, const std::optional<T> & default_value,
    bool log_missing = true ) noexcept
{
    using namespace Utility;
    const auto node = tbl.at_path( key );
    if( !node )
    {
        if( default_value )
        {
            dest = *default_value;
            Log( Log_Level::Info, Log_Sender::IO,
                 fmt::format( "Key '{}' was set to global default: {}", key, detail::pre_format( *default_value ) ) );
            return;
        }

        if( log_missing )
            Log( Log_Level::Warning, Log_Sender::IO,
                 fmt::format( "Missing key encountererd!: '{}', using default: {}", key, detail::pre_format( dest ) ) );
        return;
    }

    if( auto result = toml_transform<T>( *node.node() ) )
        dest = *result;
    else
        Log( Log_Level::Error, Log_Sender::IO,
             fmt::format(
                 "Failed converting value for key: '{}', using default: {}", key, detail::pre_format( dest ) ) );
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
