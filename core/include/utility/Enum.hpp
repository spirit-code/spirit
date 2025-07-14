#pragma once
#include <optional>
#include <string_view>
#include <tuple>

namespace Utility
{

namespace Enum
{

// table element format: { enum_value, repr, short_name, full_name }
template<typename Enum>
using TableElementType = std::tuple<Enum, std::string_view, std::string_view, std::string_view>;

// short table element format: { enum_value, repr, short_name, full_name=short_name }
template<typename Enum>
using ShortTableElementType = std::tuple<Enum, std::string_view, std::string_view>;

inline constexpr std::string_view unknown = "<unknown>";

template<typename T>
constexpr auto from_string( const std::string_view string ) -> std::optional<T>
{
    static_assert( std::is_enum_v<T> );
    for( const auto & entry : enum_table( T{} ) )
        if( std::get<1>( entry ) == string )
            return { std::get<0>( entry ) };
    return std::nullopt;
}

template<typename T>
constexpr std::string_view to_string( const T enum_value )
{
    static_assert( std::is_enum_v<T> );
    // constexpr compatible linear search
    for( const auto & entry : enum_table( T{} ) )
        if( std::get<0>( entry ) == enum_value )
            return std::get<1>( entry );
    return unknown;
}

template<typename T>
constexpr std::string_view name( const T enum_value )
{
    static_assert( std::is_enum_v<T> );
    // constexpr compatible linear search
    for( const auto & entry : enum_table( T{} ) )
        if( std::get<0>( entry ) == enum_value )
            return std::get<2>( entry );
    return unknown;
}

template<typename T>
constexpr std::string_view full_name( const T enum_value )
{
    static_assert( std::is_enum_v<T> );
    // constexpr compatible linear search
    for( const auto & entry : enum_table( T{} ) )
        if( std::get<0>( entry ) == enum_value )
        {
            // support for the short table format
            if constexpr( 3 == std::tuple_size_v<std::decay_t<decltype( entry )>> )
                return std::get<2>( entry );
            else
                return std::get<3>( entry );
        }
    return unknown;
}

template<typename T>
constexpr auto is_valid( T enum_value ) -> bool
{
    static_assert( std::is_enum_v<T> );
    for( const auto & entry : enum_table( T{} ) )
        if( std::get<0>( entry ) == enum_value )
            return true;
    return false;
}

template<typename T, typename Integral>
constexpr auto from_integral( Integral i ) -> std::optional<T>
{
    static_assert( std::is_enum_v<T> );
    static_assert( std::is_integral_v<Integral> );

    T enum_value = static_cast<T>( i );
    if( Enum::is_valid( enum_value ) )
        return std::optional{ enum_value };
    else
        return std::nullopt;
};

template<typename Integral = int, typename T>
constexpr auto to_integral( T enum_value ) -> Integral
{
    static_assert( std::is_enum_v<T> );
    static_assert( std::is_integral_v<Integral> );

    return static_cast<Integral>( enum_value );
};

} // namespace Enum

} // namespace Utility
