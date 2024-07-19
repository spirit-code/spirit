#pragma once

#if defined( SPIRIT_USE_OPENMP ) && !defined( SPIRIT_USE_STDPAR )
#include <engine/backend/Functor.hpp>
#include <engine/backend/types.hpp>

#include <iterator>

namespace Engine
{

namespace Backend
{

namespace cpu
{

namespace detail
{

template<typename ForwardIt, typename Size>
auto reduce_n( const ::execution::par_t &, ForwardIt first, Size n ) ->
    typename std::iterator_traits<ForwardIt>::value_type
{
    typename std::iterator_traits<ForwardIt>::value_type init = 0;
#pragma omp parallel for reduction( + : init )
    for( Size i = 0; i < n; ++i )
    {
        init += *( first + i );
    }
    return init;
};

template<typename ForwardIt, typename Size, typename T>
auto reduce_n( const ::execution::par_t &, ForwardIt first, Size n, T init, Backend::max<T> ) -> T
{
#pragma omp parallel for reduction( max : init )
    for( Size i = 0; i < n; ++i )
    {
        init = std::max( init, *( first + i ) );
    }
    return init;
};

template<typename ForwardIt, typename Size, typename T>
auto reduce_n( const ::execution::par_t &, ForwardIt first, Size n, T init, Backend::plus<T> ) -> T
{
#pragma omp parallel for reduction( + : init )
    for( Size i = 0; i < n; ++i )
    {
        init += *( first + i );
    }
    return init;
};

template<typename ForwardIt, typename Size, typename T, typename BinaryReductionOp>
auto reduce_n( const ::execution::par_t &, ForwardIt first, Size n, T init, BinaryReductionOp binary_op ) -> T
{
#pragma omp parallel
    {
        T local_result = zero_value<T>();
#pragma omp for
        for( Size i = 0; i < n; ++i )
        {
            local_result = binary_op( local_result, *( first + i ) );
        }
#pragma critical
        init = binary_op( init, local_result );
    }
    return init;
};

template<typename ForwardIt1, typename Size, typename ForwardIt2, typename UnaryOp>
ForwardIt2 transform_n( const ::execution::par_t &, ForwardIt1 first, Size n, ForwardIt2 d_first, UnaryOp unary_op )
{
#pragma omp parallel for
    for( Size i = 0; i < n; ++i )
    {
        *( d_first + i ) = unary_op( *( first + i ) );
    }
    return std::next( d_first, n );
}

template<typename ForwardIt1, typename Size, typename ForwardIt2, typename ForwardIt3, typename BinaryOp>
ForwardIt3 transform_n(
    const ::execution::par_t &, ForwardIt1 first1, Size n, ForwardIt2 first2, ForwardIt3 d_first, BinaryOp binary_op )
{
#pragma omp parallel for
    for( Size i = 0; i < n; ++i )
    {
        *( d_first + i ) = binary_op( *( first1 + i ), *( first2 + i ) );
    }
    return std::next( d_first, n );
}

template<typename ForwardIt, typename Size, typename T, typename UnaryOp>
T transform_reduce_n( const ::execution::par_t &, ForwardIt first, Size n, T init, Backend::max<T>, UnaryOp unary_op )
{
#pragma omp parallel for reduction( max : init )
    for( Size i = 0; i < n; ++i )
    {
        init = std::max( init, unary_op( *( first + i ) ) );
    }
    return init;
}

template<typename ForwardIt, typename Size, typename T, typename UnaryOp>
T transform_reduce_n( const ::execution::par_t &, ForwardIt first, Size n, T init, Backend::plus<T>, UnaryOp unary_op )
{
#pragma omp parallel for reduction( + : init )
    for( Size i = 0; i < n; ++i )
    {
        init += unary_op( *( first + i ) );
    }
    return init;
}

template<typename ForwardIt, typename Size, typename T, typename BinaryReductionOp, typename UnaryOp>
T transform_reduce_n(
    const ::execution::par_t &, ForwardIt first, Size n, T init, BinaryReductionOp reduce, UnaryOp unary_op )
{
#pragma omp parallel
    {
        T local_result = zero_value<T>();
#pragma omp for
        for( Size i = 0; i < n; ++i )
        {
            local_result = reduce( local_result, unary_op( *( first + i ) ) );
        }
#pragma critical
        init = reduce( init, local_result );
    }
    return init;
}

template<typename ForwardIt1, typename Size, typename ForwardIt2, typename T, typename BinaryOp>
T transform_reduce_n(
    const ::execution::par_t &, ForwardIt1 first1, Size n, ForwardIt2 first2, T init, Backend::max<T>,
    BinaryOp binary_op )
{
#pragma omp parallel for reduction( max : init )
    for( Size i = 0; i < n; ++i )
    {
        init = std::max( init, binary_op( *( first1 + i ), *( first2 + i ) ) );
    }
    return init;
}

template<typename ForwardIt1, typename Size, typename ForwardIt2, typename T, typename BinaryOp>
T transform_reduce_n(
    const ::execution::par_t &, ForwardIt1 first1, Size n, ForwardIt2 first2, T init, Backend::plus<T>,
    BinaryOp binary_op )
{
#pragma omp parallel for reduction( + : init )
    for( Size i = 0; i < n; ++i )
    {
        init += binary_op( *( first1 + i ), *( first2 + i ) );
    }
    return init;
}

template<
    typename ForwardIt1, typename Size, typename ForwardIt2, typename T, typename BinaryReductionOp, typename BinaryOp>
T transform_reduce_n(
    const ::execution::par_t &, ForwardIt1 first1, Size n, ForwardIt2 first2, T init, BinaryReductionOp reduce,
    BinaryOp binary_op )
{
#pragma omp parallel
    {
        T local_result = zero_value<T>();
#pragma omp for
        for( Size i = 0; i < n; ++i )
        {
            local_result = reduce( local_result, binary_op( *( first1 + i ), *( first2 + i ) ) );
        }
#pragma critical
        init = reduce( init, local_result );
    }
    return init;
}

} // namespace detail

template<typename ForwardIt1, typename Size, typename ForwardIt2>
ForwardIt2 copy_n( const ::execution::par_t &, ForwardIt1 first, Size n, ForwardIt2 d_first )
{
#pragma omp parallel for
    for( Size i = 0; i < n; ++i )
    {
        *( d_first + i ) = *( first + i );
    }
    return std::next( d_first, n );
}

template<typename ForwardIt, typename Size, typename T>
void fill_n( const ::execution::par_t &, ForwardIt first, Size n, const T & value )
{
#pragma omp parallel for
    for( Size i = 0; i < n; ++i )
    {
        *( first + i ) = value;
    }
}

template<typename ForwardIt, typename Size, typename UnaryOp>
void for_each_n( const ::execution::par_t &, ForwardIt first, Size n, UnaryOp unary_op )
{
#pragma omp parallel for
    for( Size i = 0; i < n; ++i )
    {
        unary_op( *( first + i ) );
    }
}

template<typename ForwardIt1, typename ForwardIt2>
ForwardIt2 copy( const ::execution::par_t & exec, ForwardIt1 first, ForwardIt1 last, ForwardIt2 d_first )
{
    return Backend::cpu::copy_n( exec, first, std::distance( first, last ), d_first );
}

template<typename ForwardIt, typename T>
void fill( const ::execution::par_t & exec, ForwardIt first, ForwardIt last, const T & value )
{
    return Backend::cpu::fill_n( exec, first, std::distance( first, last ), value );
}

template<typename ForwardIt, typename UnaryOp>
void for_each( const ::execution::par_t & exec, ForwardIt first, ForwardIt last, UnaryOp unary_op )
{
    Backend::cpu::for_each_n( exec, first, std::distance( first, last ), unary_op );
}

template<typename ForwardIt>
auto reduce( const ::execution::par_t & exec, ForwardIt first, ForwardIt last ) ->
    typename std::iterator_traits<ForwardIt>::value_type
{
    return Backend::cpu::detail::reduce_n( exec, first, std::distance( first, last ) );
};

template<typename ForwardIt, typename T, typename BinaryOp>
auto reduce( const ::execution::par_t & exec, ForwardIt first, ForwardIt last, T init, BinaryOp binary_op ) ->
    typename std::iterator_traits<ForwardIt>::value_type
{
    return Backend::cpu::detail::reduce_n( exec, first, std::distance( first, last ), init, binary_op );
};

template<typename ForwardIt1, typename ForwardIt2, typename UnaryOp>
ForwardIt2
transform( const ::execution::par_t & exec, ForwardIt1 first, ForwardIt1 last, ForwardIt2 d_first, UnaryOp unary_op )
{
    return Backend::cpu::detail::transform_n( exec, first, std::distance( first, last ), d_first, unary_op );
}

template<typename ForwardIt1, typename ForwardIt2, typename ForwardIt3, typename BinaryOp>
ForwardIt3 transform(
    const ::execution::par_t & exec, ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2, ForwardIt3 d_first,
    BinaryOp binary_op )
{
    return Backend::cpu::detail::transform_n(
        exec, first1, std::distance( first1, last1 ), first2, d_first, binary_op );
}

template<typename ForwardIt, typename T, typename BinaryReductionOp, typename UnaryOp>
T transform_reduce(
    const ::execution::par_t & exec, ForwardIt first, ForwardIt last, T init, BinaryReductionOp reduce,
    UnaryOp unary_op )
{
    return Backend::cpu::detail::transform_reduce_n(
        exec, first, std::distance( first, last ), init, reduce, unary_op );
}

template<typename ForwardIt1, typename ForwardIt2, typename T, typename BinaryReductionOp, typename BinaryOp>
T transform_reduce(
    const ::execution::par_t & exec, ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2, T init,
    BinaryReductionOp reduce, BinaryOp binary_op )
{
    return Backend::cpu::detail::transform_reduce_n(
        exec, first1, std::distance( first1, last1 ), first2, init, reduce, binary_op );
}

} // namespace cpu

} // namespace Backend

} // namespace Engine

#endif
