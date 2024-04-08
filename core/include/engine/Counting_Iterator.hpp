#pragma once

#include <engine/Vectormath_Defines.hpp>

#ifdef SPIRIT_USE_CUDA
#include <thrust/iterator/counting_iterator.h>
#endif

namespace Engine
{

namespace Backend
{

#ifndef SPIRIT_USE_CUDA
template<typename T>
class counting_iterator
{
public:
    using value_type        = T;
    using difference_type   = ptrdiff_t;
    using pointer           = T *;
    using reference         = T &;
    using iterator_category = std::random_access_iterator_tag;

    static_assert( std::is_integral<T>::value );
    static_assert( std::is_convertible<value_type, difference_type>::value );

    constexpr explicit counting_iterator( T value ) noexcept : m_value( value ){};

    constexpr value_type operator*() const noexcept
    {
        return m_value;
    }
    constexpr value_type operator[]( difference_type n ) const noexcept
    {
        return m_value + n;
    }

    constexpr counting_iterator & operator++() noexcept
    {
        ++m_value;
        return *this;
    }

    constexpr counting_iterator operator++( int ) const noexcept
    {
        counting_iterator tmp( *this );
        ++( *this );
        return tmp;
    }

    constexpr counting_iterator & operator--() noexcept
    {
        --m_value;
        return *this;
    }

    constexpr counting_iterator operator--( int ) const noexcept
    {
        counting_iterator tmp( *this );
        --( *this );
        return tmp;
    }

    constexpr counting_iterator operator+( T n ) const noexcept
    {
        return counting_iterator( m_value + n );
    }

    constexpr counting_iterator & operator+=( T n ) noexcept
    {
        m_value += n;
        return *this;
    }

    constexpr counting_iterator operator-( difference_type n ) const noexcept
    {
        return counting_iterator( m_value - n );
    }
    template<typename U>
    constexpr counting_iterator operator-( const counting_iterator<U> & other ) const noexcept
    {
        return static_cast<difference_type>( m_value - other.m_value );
    }

    constexpr counting_iterator & operator-=( T n ) noexcept
    {
        m_value -= n;
        return *this;
    }

    template<typename U, typename V>
    friend constexpr bool operator==( const counting_iterator<U> & lhs, const counting_iterator<V> & rhs ) noexcept
    {
        return lhs.m_value == rhs.m_value;
    };

    template<typename U, typename V>
    friend constexpr bool operator!=( const counting_iterator<U> & lhs, const counting_iterator<V> & rhs ) noexcept
    {
        return lhs.m_value != rhs.m_value;
    };

    template<typename U, typename V>
    friend constexpr bool operator<( const counting_iterator<U> & lhs, const counting_iterator<V> & rhs ) noexcept
    {
        return lhs.m_value < rhs.m_value;
    };

    template<typename U, typename V>
    friend constexpr bool operator<=( const counting_iterator<U> & lhs, const counting_iterator<V> & rhs ) noexcept
    {
        return lhs.m_value <= rhs.m_value;
    };

    template<typename U, typename V>
    friend constexpr bool operator>( const counting_iterator<U> & lhs, const counting_iterator<V> & rhs ) noexcept
    {
        return lhs.m_value > rhs.m_value;
    };

    template<typename U, typename V>
    friend constexpr bool operator>=( const counting_iterator<U> & lhs, const counting_iterator<V> & rhs ) noexcept
    {
        return lhs.m_value >= rhs.m_value;
    };

private:
    T m_value;
};

template<typename T>
counting_iterator<T> make_counting_iterator( T value )
{
    return counting_iterator<T>( value );
}

#else

// `thrust::for_each_n` refuses to do anything with our `host` `device` enabled `counting_iterator`, so we use theirs
// for the cuda backend
using thrust::counting_iterator;
using thrust::make_counting_iterator;

#endif

} // namespace Backend

} // namespace Engine
