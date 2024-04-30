#pragma once

#include <engine/Vectormath_Defines.hpp>

#include <iterator>
#include <type_traits>

namespace Engine
{

namespace Backend
{

namespace cuda
{

// transform iterator (similar to `cub::TransformInputIterator`)
// NOTE: This iterator assumes that the passed in iterator is a random access iterator and that both the iterator and
//       the unary_op are __host__ __device__ enabled
// TODO: write a proper `Backend::cpu::transform_iterator` implementation that doesn't need those assumptions avoidung
//       code duplucation
template<typename Value, typename UnaryOp, typename InputIt>
class transform_iterator
{
public:
    static_assert( std::is_copy_constructible<UnaryOp>::value );
    static_assert( std::is_base_of<
                   std::random_access_iterator_tag, typename std::iterator_traits<InputIt>::iterator_category>::value );
    using value_type        = Value;
    using pointer           = Value *;
    using reference         = Value;
    using difference_type   = typename std::iterator_traits<InputIt>::difference_type;
    using iterator_category = typename std::iterator_traits<InputIt>::iterator_category;

    SPIRIT_HOSTDEVICE constexpr transform_iterator( InputIt iter, UnaryOp unary_op ) noexcept
            : m_iter( std::move( iter ) ), unary_op( std::move( unary_op ) ){};
    SPIRIT_HOSTDEVICE constexpr transform_iterator( const transform_iterator & other ) noexcept
            : m_iter( other.m_iter ), unary_op( other.unary_op ){};
    constexpr transform_iterator & operator=( const transform_iterator & other ) noexcept = default;

    [[nodiscard]] SPIRIT_HOSTDEVICE constexpr reference operator*() noexcept
    {
        return unary_op( *m_iter );
    }
    [[nodiscard]] SPIRIT_HOSTDEVICE constexpr reference operator[]( difference_type n ) noexcept
    {
        return unary_op( m_iter[n] );
    }

    SPIRIT_HOSTDEVICE constexpr transform_iterator & operator++() noexcept
    {
        ++m_iter;
        return *this;
    }

    SPIRIT_HOSTDEVICE constexpr transform_iterator operator++( int ) noexcept
    {
        transform_iterator tmp( *this );
        ++( *this );
        return tmp;
    }

    SPIRIT_HOSTDEVICE constexpr transform_iterator & operator--() noexcept
    {
        --m_iter;
        return *this;
    }

    SPIRIT_HOSTDEVICE constexpr transform_iterator operator--( int ) noexcept
    {
        transform_iterator tmp( *this );
        --( *this );
        return tmp;
    }

    [[nodiscard]] SPIRIT_HOSTDEVICE constexpr transform_iterator operator+( const difference_type n ) const noexcept
    {
        return transform_iterator( m_iter + n, unary_op );
    }

    SPIRIT_HOSTDEVICE constexpr transform_iterator & operator+=( const difference_type n ) noexcept
    {
        m_iter += n;
        return *this;
    }

    [[nodiscard]] SPIRIT_HOSTDEVICE constexpr transform_iterator operator-( const difference_type n ) const noexcept
    {
        return transform_iterator( m_iter - n, unary_op );
    }

    [[nodiscard]] SPIRIT_HOSTDEVICE constexpr difference_type
    operator-( const transform_iterator & other ) const noexcept
    {
        return static_cast<difference_type>( m_iter - other.m_iter );
    }

    SPIRIT_HOSTDEVICE constexpr transform_iterator & operator-=( const difference_type n ) noexcept
    {
        m_iter -= n;
        return *this;
    }

    [[nodiscard]] friend SPIRIT_HOSTDEVICE constexpr bool
    operator==( const transform_iterator & lhs, const transform_iterator & rhs ) noexcept
    {
        return lhs.m_iter == rhs.m_iter;
    };

    [[nodiscard]] friend SPIRIT_HOSTDEVICE constexpr bool
    operator!=( const transform_iterator & lhs, const transform_iterator & rhs ) noexcept
    {
        return lhs.m_iter != rhs.m_iter;
    };

    [[nodiscard]] friend SPIRIT_HOSTDEVICE constexpr bool
    operator<( const transform_iterator & lhs, const transform_iterator & rhs ) noexcept
    {
        return lhs.m_iter < rhs.m_iter;
    };

    [[nodiscard]] friend SPIRIT_HOSTDEVICE constexpr bool
    operator<=( const transform_iterator & lhs, const transform_iterator & rhs ) noexcept
    {
        return lhs.m_iter <= rhs.m_iter;
    };

    [[nodiscard]] friend SPIRIT_HOSTDEVICE constexpr bool
    operator>( const transform_iterator & lhs, const transform_iterator & rhs ) noexcept
    {
        return lhs.m_iter > rhs.m_iter;
    };

    [[nodiscard]] friend SPIRIT_HOSTDEVICE constexpr bool
    operator>=( const transform_iterator & lhs, const transform_iterator & rhs ) noexcept
    {
        return lhs.m_iter >= rhs.m_iter;
    };

private:
    InputIt m_iter;
    UnaryOp unary_op;
};

template<
    typename UnaryOp, typename InputIt,
    typename Value = decltype( std::declval<UnaryOp>()( *std::declval<InputIt>() ) )>
[[nodiscard]] SPIRIT_HOSTDEVICE constexpr auto make_transform_iterator( InputIt iter, UnaryOp unary_op )
    -> transform_iterator<Value, UnaryOp, InputIt>
{
    return transform_iterator<Value, UnaryOp, InputIt>( iter, unary_op );
}

} // namespace cuda

namespace cpu
{

using Backend::cuda::make_transform_iterator;
using Backend::cuda::transform_iterator;

} // namespace cpu

using Backend::cuda::make_transform_iterator;
using Backend::cuda::transform_iterator;

} // namespace Backend

} // namespace Engine
