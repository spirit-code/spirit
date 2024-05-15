#pragma once

#include <type_traits>

namespace Utility
{

template<typename T, typename Exp = unsigned int>
constexpr T fastpow( T base, Exp exp ) noexcept
{
    static_assert( std::is_integral<Exp>::value, "Exponent type must be integral!" );
    static_assert( !std::is_signed<Exp>::value, "Exponent type must be unsigned!" );

    T result = 1.0;
    while( exp != 0u )
    {
        if( ( exp % 2u ) != 0u )
            result *= base;
        exp /= 2u;
        base *= base;
    }
    return result;
}

}
