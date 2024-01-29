#include <fmt/ranges.h>

#include <array>
#include <cstddef>
#include <tuple>

// cast a variable size C++ 2D-array to a C-style 2D-array
template<typename T, std::size_t N = std::tuple_size_v<typename T::value_type>>
static inline decltype( auto ) array_cast( T & v )
{
    using array_type = typename T::value_type;
    using value_type = typename array_type::value_type;
    static_assert(
        std::is_same_v<std::decay_t<array_type>, std::array<value_type, std::tuple_size_v<array_type>>>,
        "T must be a container containing std::array" );
    if constexpr( std::is_const_v<T> || std::is_const_v<array_type> || std::is_const_v<value_type> )
        return reinterpret_cast<const value_type( * )[N]>( v.data()->data() );
    else
        return reinterpret_cast<value_type( * )[N]>( v.data()->data() );
};

template<typename T, std::size_t N>
constexpr inline decltype( auto ) array_fmt( const std::array<T, N> & arr, const char * sep = ", " )
{
    return fmt::join( cbegin( arr ), cend( arr ), sep );
}

// // ostream overload to use for std::array
// template<typename T, std::size_t N>
// std::ostream & operator<<( std::ostream && os, const std::array<T, N> & arr )
// {
//     if constexpr( N == 0 )
//     {
//         return os << "{ }";
//     }
//     else
//     {
//         os << "{ ";
//         std::for_each_n( begin( arr ), N - 1, [&os]( const T & element ) { os << element << ", "; } );
//         return os << arr.back() << " }";
//     }
// }
//
// template<typename T, std::size_t N>
// struct fmt::formatter<std::array<T, N>, char> : ostream_formatter
// {
// };
