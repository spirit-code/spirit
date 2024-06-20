#pragma once

#include <engine/Vectormath_Defines.hpp>
#include <engine/backend/types.hpp>
#include <iterator>

namespace Engine
{

namespace Backend
{

namespace cuda
{

// zip iterator (similar to `thrust::zip_function`)
template<typename Function>
class zip_function
{
public:
    explicit constexpr zip_function( Function func ) noexcept( std::is_nothrow_move_constructible<Function>::value )
            : func( std::move( func ) ){};
    explicit constexpr zip_function( Function && func ) noexcept( std::is_nothrow_move_constructible<Function>::value )
            : func( std::move( func ) ){};

    template<typename Tuple>
    SPIRIT_HOSTDEVICE constexpr auto operator()( Tuple && args ) -> decltype( auto )
    {
        return Backend::apply( func, Backend::forward<Tuple>( args ) );
    };

private:
    Function func;
};

template<typename Functor>
SPIRIT_HOSTDEVICE auto make_zip_function( Functor f ) -> zip_function<Functor>
{
    return zip_function<Functor>( f );
}

template<typename IteratorTuple>
class zip_iterator;

// zip iterator (similar to `thrust::zip_iterator`)
// NOTE: This iterator assumes that all passed in iterators are random access iterators that are __host__ __device__ enabled
// TODO: write a proper `Backend::cpu::zip_iterator` implementation that doesn't need those assumptions (while avoidung
//       code duplucation)
template<typename... Iterators>
class zip_iterator<Backend::tuple<Iterators...>>
{
    using IteratorTuple = Backend::tuple<Iterators...>;

public:
    static_assert(
        std::conjunction<std::is_base_of<
            std::random_access_iterator_tag, typename std::iterator_traits<Iterators>::iterator_category>...>::value,
        "All passed in iterators must at least be random access iterators" );

    using value_type = Backend::tuple<typename std::iterator_traits<Iterators>::value_type...>;
    using pointer    = Backend::tuple<typename std::iterator_traits<Iterators>::pointer...>;
    using reference  = Backend::tuple<typename std::iterator_traits<Iterators>::reference...>;
    using difference_type =
        typename std::iterator_traits<typename std::tuple_element<0, IteratorTuple>::type>::difference_type;
    using iterator_category = std::random_access_iterator_tag;

    SPIRIT_HOSTDEVICE constexpr zip_iterator( IteratorTuple iterators ) noexcept(
        std::is_nothrow_constructible<IteratorTuple>::value )
            : m_state( std::move( iterators ) ){};

    SPIRIT_HOSTDEVICE constexpr zip_iterator( const zip_iterator & other ) noexcept(
        std::is_nothrow_copy_constructible<IteratorTuple>::value )
            : m_state( other.m_state ){};
    constexpr zip_iterator &
    operator=( const zip_iterator & other ) noexcept( std::is_nothrow_copy_assignable<IteratorTuple>::value )
        = default;

    [[nodiscard]] SPIRIT_HOSTDEVICE constexpr reference operator*() noexcept
    {
        return Backend::apply( [] SPIRIT_HOSTDEVICE( Iterators & ... it ) { return reference( *it... ); }, m_state );
    }

    [[nodiscard]] SPIRIT_HOSTDEVICE constexpr reference operator[]( difference_type n ) noexcept
    {
        return Backend::apply( [n] SPIRIT_HOSTDEVICE( Iterators & ... it ) { return reference( it[n]... ); }, m_state );
    }

    SPIRIT_HOSTDEVICE constexpr zip_iterator & operator++() noexcept
    {
        Backend::apply( [] SPIRIT_HOSTDEVICE( Iterators & ... it ) { ( ++it, ... ); }, m_state );
        return *this;
    }

    SPIRIT_HOSTDEVICE constexpr zip_iterator
    operator++( int ) noexcept( std::is_nothrow_copy_constructible<zip_iterator>::value )
    {
        zip_iterator tmp( *this );
        ++( *this );
        return tmp;
    }

    SPIRIT_HOSTDEVICE constexpr zip_iterator & operator--() noexcept
    {
        Backend::apply( [] SPIRIT_HOSTDEVICE( Iterators & ... it ) { ( --it, ... ); }, m_state );
        return *this;
    }

    SPIRIT_HOSTDEVICE constexpr zip_iterator
    operator--( int ) noexcept( std::is_nothrow_copy_constructible<zip_iterator>::value )
    {
        zip_iterator tmp( *this );
        --( *this );
        return tmp;
    }

    [[nodiscard]] SPIRIT_HOSTDEVICE constexpr zip_iterator operator+( const difference_type n ) const noexcept
    {
        return Backend::apply(
            [n] SPIRIT_HOSTDEVICE( const Iterators &... it )
            { return zip_iterator( Backend::make_tuple( ( it + n )... ) ); },
            m_state );
    }

    SPIRIT_HOSTDEVICE constexpr zip_iterator & operator+=( const difference_type n ) noexcept
    {
        Backend::apply( [n] SPIRIT_HOSTDEVICE( Iterators & ... it ) { ( ( it += n ), ... ); }, m_state );
        return *this;
    }

    [[nodiscard]] SPIRIT_HOSTDEVICE constexpr zip_iterator operator-( const difference_type n ) const noexcept
    {
        return Backend::apply(
            [n] SPIRIT_HOSTDEVICE( const Iterators &... it )
            { return zip_iterator( Backend::make_tuple( ( it - n )... ) ); },
            m_state );
    }

    [[nodiscard]] SPIRIT_HOSTDEVICE constexpr difference_type operator-( const zip_iterator & other ) const noexcept
    {
        return static_cast<difference_type>( Backend::get<0>( m_state ) - Backend::get<0>( other.m_state ) );
    }

    SPIRIT_HOSTDEVICE constexpr zip_iterator & operator-=( const difference_type n ) noexcept
    {
        Backend::apply( [n] SPIRIT_HOSTDEVICE( Iterators & ... it ) { ( ( it -= n ), ... ); }, m_state );
        return *this;
    }

    SPIRIT_HOSTDEVICE constexpr IteratorTuple & get() noexcept
    {
        return m_state;
    }

    [[nodiscard]] friend SPIRIT_HOSTDEVICE constexpr bool
    operator==( const zip_iterator & lhs, const zip_iterator & rhs ) noexcept
    {
        return Backend::get<0>( lhs.m_state ) == Backend::get<0>( rhs.m_state );
    };

    [[nodiscard]] friend SPIRIT_HOSTDEVICE constexpr bool
    operator!=( const zip_iterator & lhs, const zip_iterator & rhs ) noexcept
    {
        return Backend::get<0>( lhs.m_state ) != Backend::get<0>( rhs.m_state );
    };

    [[nodiscard]] friend SPIRIT_HOSTDEVICE constexpr bool
    operator<( const zip_iterator & lhs, const zip_iterator & rhs ) noexcept
    {
        return Backend::get<0>( lhs.m_state ) < Backend::get<0>( rhs.m_state );
    };

    [[nodiscard]] friend SPIRIT_HOSTDEVICE constexpr bool
    operator<=( const zip_iterator & lhs, const zip_iterator & rhs ) noexcept
    {
        return Backend::get<0>( lhs.m_state ) <= Backend::get<0>( rhs.m_state );
    };

    [[nodiscard]] friend SPIRIT_HOSTDEVICE constexpr bool
    operator>( const zip_iterator & lhs, const zip_iterator & rhs ) noexcept
    {
        return Backend::get<0>( lhs.m_state ) > Backend::get<0>( rhs.m_state );
    };

    [[nodiscard]] friend SPIRIT_HOSTDEVICE constexpr bool
    operator>=( const zip_iterator & lhs, const zip_iterator & rhs ) noexcept
    {
        return Backend::get<0>( lhs.m_state ) >= Backend::get<0>( rhs.m_state );
    };

private:
    IteratorTuple m_state;
};

template<typename... Iterators>
[[nodiscard]] SPIRIT_HOSTDEVICE constexpr auto make_zip_iterator( Backend::tuple<Iterators...> tuple )
    -> zip_iterator<Backend::tuple<Iterators...>>
{
    return zip_iterator<Backend::tuple<Iterators...>>( tuple );
}

template<typename... Iterators>
[[nodiscard]] SPIRIT_HOSTDEVICE constexpr auto make_zip_iterator( Iterators... iter )
    -> zip_iterator<Backend::tuple<Iterators...>>
{
    return zip_iterator<Backend::tuple<Iterators...>>( Backend::make_tuple( iter... ) );
}

} // namespace cuda

namespace cpu
{

using Backend::cuda::make_zip_function;
using Backend::cuda::make_zip_iterator;
using Backend::cuda::zip_iterator;
using Backend::cuda::zip_function;

} // namespace cpu

using Backend::cuda::make_zip_function;
using Backend::cuda::make_zip_iterator;
using Backend::cuda::zip_iterator;
using Backend::cuda::zip_function;

} // namespace Backend

} // namespace Engine
