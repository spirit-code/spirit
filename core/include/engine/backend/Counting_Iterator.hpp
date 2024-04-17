#pragma once

#include <engine/Vectormath_Defines.hpp>

#include <type_traits>
#include <iterator>

namespace Engine
{

namespace Backend
{

template<typename T>
class counting_iterator
{
public:
    using value_type        = T;
    using difference_type   = std::ptrdiff_t;
    using pointer           = T *;
    using reference         = T &;
    using iterator_category = std::random_access_iterator_tag;

    static_assert( std::is_integral<T>::value );
    static_assert( std::is_convertible<value_type, difference_type>::value );

    SPIRIT_HOSTDEVICE constexpr counting_iterator() noexcept : m_value( T{} ){};
    SPIRIT_HOSTDEVICE constexpr explicit counting_iterator( T value ) noexcept : m_value( value ){};
    SPIRIT_HOSTDEVICE constexpr counting_iterator( const counting_iterator & other ) noexcept
            : m_value( other.m_value ){};
    constexpr counting_iterator & operator=( const counting_iterator & other ) noexcept = default;

    [[nodiscard]] SPIRIT_HOSTDEVICE constexpr value_type operator*() const noexcept
    {
        return m_value;
    }
    [[nodiscard]] SPIRIT_HOSTDEVICE constexpr value_type operator[]( difference_type n ) const noexcept
    {
        return m_value + n;
    }

    SPIRIT_HOSTDEVICE constexpr counting_iterator & operator++() noexcept
    {
        ++m_value;
        return *this;
    }

    SPIRIT_HOSTDEVICE constexpr counting_iterator operator++( int ) noexcept
    {
        counting_iterator tmp( *this );
        ++( *this );
        return tmp;
    }

    SPIRIT_HOSTDEVICE constexpr counting_iterator & operator--() noexcept
    {
        --m_value;
        return *this;
    }

    SPIRIT_HOSTDEVICE constexpr counting_iterator operator--( int ) noexcept
    {
        counting_iterator tmp( *this );
        --( *this );
        return tmp;
    }

    [[nodiscard]] SPIRIT_HOSTDEVICE constexpr counting_iterator operator+( T n ) const noexcept
    {
        return counting_iterator( m_value + n );
    }

    SPIRIT_HOSTDEVICE constexpr counting_iterator & operator+=( T n ) noexcept
    {
        m_value += n;
        return *this;
    }

    [[nodiscard]] SPIRIT_HOSTDEVICE constexpr counting_iterator operator-( difference_type n ) const noexcept
    {
        return counting_iterator( m_value - n );
    }

    template<typename U>
    [[nodiscard]] SPIRIT_HOSTDEVICE constexpr difference_type
    operator-( const counting_iterator<U> & other ) const noexcept
    {
        return static_cast<difference_type>( m_value - other.m_value );
    }

    SPIRIT_HOSTDEVICE constexpr counting_iterator & operator-=( T n ) noexcept
    {
        m_value -= n;
        return *this;
    }

    template<typename U, typename V>
    [[nodiscard]] friend SPIRIT_HOSTDEVICE constexpr bool
    operator==( const counting_iterator<U> & lhs, const counting_iterator<V> & rhs ) noexcept
    {
        return lhs.m_value == rhs.m_value;
    };

    template<typename U, typename V>
    [[nodiscard]] friend SPIRIT_HOSTDEVICE constexpr bool
    operator!=( const counting_iterator<U> & lhs, const counting_iterator<V> & rhs ) noexcept
    {
        return lhs.m_value != rhs.m_value;
    };

    template<typename U, typename V>
    [[nodiscard]] friend SPIRIT_HOSTDEVICE constexpr bool
    operator<( const counting_iterator<U> & lhs, const counting_iterator<V> & rhs ) noexcept
    {
        return lhs.m_value < rhs.m_value;
    };

    template<typename U, typename V>
    [[nodiscard]] friend SPIRIT_HOSTDEVICE constexpr bool
    operator<=( const counting_iterator<U> & lhs, const counting_iterator<V> & rhs ) noexcept
    {
        return lhs.m_value <= rhs.m_value;
    };

    template<typename U, typename V>
    [[nodiscard]] friend SPIRIT_HOSTDEVICE constexpr bool
    operator>( const counting_iterator<U> & lhs, const counting_iterator<V> & rhs ) noexcept
    {
        return lhs.m_value > rhs.m_value;
    };

    template<typename U, typename V>
    [[nodiscard]] friend SPIRIT_HOSTDEVICE constexpr bool
    operator>=( const counting_iterator<U> & lhs, const counting_iterator<V> & rhs ) noexcept
    {
        return lhs.m_value >= rhs.m_value;
    };

private:
    T m_value;
};

template<typename T>
[[nodiscard]] SPIRIT_HOSTDEVICE constexpr counting_iterator<T> make_counting_iterator( T value ) noexcept
{
    return counting_iterator<T>( value );
}

} // namespace Backend

} // namespace Engine
