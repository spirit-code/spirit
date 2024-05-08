#pragma once

#include <utility/Logging.hpp>

#include <fmt/format.h>

#include <array>
#include <string_view>

namespace IO
{

enum Flag : unsigned int
{
    None             = 0x0u,
    Contributions    = 0x1u << 0u,
    Normalize_by_nos = 0x1u << 1u,
    Readability      = 0x1u << 2u,
};

constexpr std::string_view name( Flag value ) noexcept
{
    switch( value )
    {
        case Flag::None: return "None";
        case Flag::Contributions: return "Contributions";
        case Flag::Normalize_by_nos: return "Normalize_by_nos";
        case Flag::Readability: return "Readability";
        default: return "Mixed";
    }
}

// clang-format off
static constexpr auto AllFlags = std::array{
    Flag::Contributions,
    Flag::Normalize_by_nos,
    Flag::Readability
};
// clang-format on

struct Flags
{
    using value_type = unsigned int;

    constexpr Flags() noexcept = default;
    constexpr Flags( Flag flag ) noexcept : m_value( flag ){};
    constexpr Flags( value_type flags ) noexcept : m_value( flags ){};

    constexpr operator bool() const
    {
        return m_value != 0;
    }

    friend constexpr std::string_view name( const Flags & flags ) noexcept
    {
        return name( static_cast<Flag>( flags.m_value ) );
    }

    constexpr Flags operator~() const noexcept
    {
        return Flags( ~m_value );
    }

    constexpr Flags & operator&=( const Flags & rhs ) noexcept
    {
        m_value &= rhs.m_value;
        return *this;
    }
    friend constexpr Flags operator&( const Flags & lhs, const Flags & rhs ) noexcept
    {
        return Flags( lhs.m_value & rhs.m_value );
    }
    friend constexpr Flags operator&( const value_type & lhs, const Flags & rhs ) noexcept
    {
        return Flags( lhs & rhs.m_value );
    }
    friend constexpr Flags operator&( const Flags & lhs, const value_type & rhs ) noexcept
    {
        return Flags( lhs.m_value & rhs );
    }

    constexpr Flags & operator^=( const Flags & rhs ) noexcept
    {
        m_value ^= rhs.m_value;
        return *this;
    }
    friend constexpr Flags operator^( const Flags & lhs, const Flags & rhs ) noexcept
    {
        return Flags( lhs.m_value ^ rhs.m_value );
    }
    friend constexpr Flags operator^( const value_type & lhs, const Flags & rhs ) noexcept
    {
        return Flags( lhs ^ rhs.m_value );
    }
    friend constexpr Flags operator^( const Flags & lhs, const value_type & rhs ) noexcept
    {
        return Flags( lhs.m_value ^ rhs );
    }

    constexpr Flags & operator|=( const Flags & rhs ) noexcept
    {
        m_value |= rhs.m_value;
        return *this;
    }
    friend constexpr Flags operator|( const Flags & lhs, const Flags & rhs ) noexcept
    {
        return Flags( lhs.m_value | rhs.m_value );
    }
    friend constexpr Flags operator|( const value_type & lhs, const Flags & rhs ) noexcept
    {
        return Flags( lhs | rhs.m_value );
    }
    friend constexpr Flags operator|( const Flags & lhs, const value_type & rhs ) noexcept
    {
        return Flags( lhs.m_value | rhs );
    }

    friend constexpr bool operator==( const Flags & lhs, const Flags & rhs ) noexcept
    {
        return lhs.m_value == rhs.m_value;
    }

    friend constexpr bool operator!=( const Flags & lhs, const Flags & rhs ) noexcept
    {
        return lhs.m_value != rhs.m_value;
    }

private:
    value_type m_value = Flag::None;
};

inline bool
verify_flags( const Flags & flags, const Flags & supported_flags = Flag::None, const std::string_view method_name = "" )
{
    const auto unsupported_flags = flags & ~supported_flags;

    if( unsupported_flags )
        return true;

    std::vector<std::string> output;
    output.reserve( AllFlags.size() + 1 );
    output.emplace_back( fmt::format( "Flags not recognized by method \"{}\":", method_name ) );
    for( const auto & flag : AllFlags )
        if( unsupported_flags & flag )
            output.emplace_back( fmt::format( "  Flag{{{}}}", name( flag ) ) );

    Log( Utility::Log_Level::Debug, Utility::Log_Sender::IO, output );
    return false;
}

} // namespace IO
