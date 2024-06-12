#pragma once

#include <array>
#include <functional>

#include <fmt/core.h>
#include <fmt/format.h>

struct Pair
{
    int i, j;
    std::array<int, 3> translations;
};

inline bool operator==( const Pair & lhs, const Pair & rhs )
{
    return std::tie( lhs.i, lhs.j, lhs.translations ) == std::tie( rhs.i, rhs.j, rhs.translations );
}

inline bool operator!=( const Pair & lhs, const Pair & rhs )
{
    return std::tie( lhs.i, lhs.j, lhs.translations ) != std::tie( rhs.i, rhs.j, rhs.translations );
}

template<typename T>
struct equivalence;

template<>
struct equivalence<Pair>
{
    constexpr bool operator()( const Pair & lhs, const Pair & rhs ) const noexcept
    {
        if( std::tie( lhs.i, lhs.j, lhs.translations ) == std::tie( rhs.i, rhs.j, rhs.translations ) )
            return true;
        if( std::tie( lhs.i, lhs.j ) != std::tie( rhs.j, rhs.i ) )
            return false;

        const auto & t = rhs.translations;
        return lhs.translations == std::array{ -t[0], -t[1], -t[2] };
    }
};

inline constexpr bool equiv( const Pair & lhs, const Pair & rhs ) noexcept
{
    return equivalence<Pair>{}( lhs, rhs );
}

inline constexpr bool parity( const Pair & pair ) noexcept
{
    if( pair.i != pair.j )
        return pair.i > pair.j;
    for( auto i = 0; i < 3; ++i )
        if( pair.translations[i] != 0 )
            return pair.translations[i] > 0;
    return true;
}

inline constexpr Pair inverse( const Pair & pair ) noexcept
{
    const auto & t = pair.translations;
    return { pair.j, pair.i, { -t[0], -t[1], -t[2] } };
}

template<typename Head, typename... Tail>
constexpr std::size_t hash_sequence( Head && head, Tail &&... tail ) noexcept
{
    if constexpr( sizeof...( Tail ) == 0 )
        return std::hash<std::decay_t<Head>>{}( std::forward<Head>( head ) );
    else
        // recursive hash combination following the python implementation for tuples
        return std::hash<std::decay_t<Head>>{}( std::forward<Head>( head ) )
               ^ ( hash_sequence( std::forward<Tail>( tail )... ) << 1u );
};

inline constexpr std::size_t hash( const Pair & pair ) noexcept
{
    return hash_sequence( pair.i, pair.j, pair.translations[0], pair.translations[1], pair.translations[2] );
}

constexpr std::size_t eqiv_hash( const Pair & pair )
{
    return hash( parity( pair ) ? pair : inverse( pair ) );
}

template<>
struct std::hash<Pair>
{
    constexpr std::size_t operator()( const Pair & pair ) const noexcept
    {
        return ::hash( pair );
    }
};

template<typename T>
struct equiv_hash;

template<>
struct equiv_hash<Pair> : std::hash<Pair>
{
    constexpr std::size_t operator()( const Pair & pair ) const noexcept
    {
        return std::hash<Pair>::operator()( parity( pair ) ? pair : inverse( pair ) );
    }
};

template<>
struct fmt::formatter<Pair> : fmt::formatter<string_view>
{
    auto format( Pair p, format_context & ctx ) const
    {
        const auto & t = p.translations;
        return fmt::formatter<string_view>::format(
            fmt::format( "Pair(i={}, j={}, t={{{}, {}, {}}})", p.i, p.j, t[0], t[1], t[2] ), ctx );
    }
};
