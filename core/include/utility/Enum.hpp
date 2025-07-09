#pragma once
#include <optional>
#include <string_view>

namespace Utility
{

namespace Enum
{

inline constexpr std::string_view unknown = "<unknown>";

template<typename T>
constexpr auto is_valid( T enum_value ) -> bool
{
    static_assert( std::is_enum_v<T> );
    return name( enum_value ).data() != unknown.data();
}

template<typename T>
constexpr auto from_string( std::string_view ) -> std::optional<T>;

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

} // namespace Enum

} // namespace Utility
