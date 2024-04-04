#pragma once
#ifndef SPIRIT_CORE_ENGINE_BACKEND_PAR_HPP
#define SPIRIT_CORE_ENGINE_BACKEND_PAR_HPP

#include <engine/Vectormath_Defines.hpp>

#include <algorithm>
#include <numeric>
#include <optional>
#include <vector>

#ifdef SPIRIT_USE_STDPAR
#include <execution>
#define SPIRIT_CPU_PAR std::execution::par,
#elseifdef SPIRIT_USE_OPENMP

namespace execution
{

struct Par
{
    constexpr Par() noexcept = default;
};

bool constexpr operator==( const Par & first, const Par & second )
{
    return true;
};
bool constexpr operator!=( const Par & first, const Par & second )
{
    return false;
};

static constexpr Par par = Par();

} // namespace execution

#define SPIRIT_CPU_PAR ::execution::par,
#else
#define SPIRIT_CPU_PAR
#endif

#ifndef SPIRIT_USE_CUDA
#define SPIRIT_PAR SPIRIT_CPU_PAR

#define SPIRIT_LAMBDA
#else
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/optional.h>
#include <thrust/transform.h>
#include <thrust/zip_function.h>

#include <cub/cub.cuh>
#include <cuda/std/tuple>

template<typename Iter>
struct is_host_iterator
        : std::conjunction<
              std::negation<std::is_pointer<Iter>>,
              std::disjunction<
                  std::is_same<Iter, typename field<typename std::iterator_traits<Iter>::value_type>::iterator>,
                  std::is_same<Iter, typename field<typename std::iterator_traits<Iter>::value_type>::const_iterator>,
                  std::is_same<Iter, typename std::vector<typename std::iterator_traits<Iter>::value_type>::iterator>,
                  std::is_same<
                      Iter, typename std::vector<typename std::iterator_traits<Iter>::value_type>::const_iterator>>>
{
};

namespace execution
{

namespace cuda
{

struct Par
{
    constexpr Par() noexcept = default;
};

bool constexpr operator==( const Par & first, const Par & second )
{
    return true;
};
bool constexpr operator!=( const Par & first, const Par & second )
{
    return false;
};

static constexpr Par par = Par();

} // namespace cuda

} // namespace execution

#define SPIRIT_PAR

#define SPIRIT_LAMBDA __device__
#endif

namespace Engine
{

namespace Backend
{

namespace cpu
{
using std::optional;
using std::vector;

using std::apply;
using std::get;
using std::make_tuple;
using std::tuple;

using std::plus;

using std::copy;
using std::fill;
using std::fill_n;
using std::for_each;
using std::transform;
using std::transform_reduce;

} // namespace cpu

#ifndef SPIRIT_USE_CUDA
using namespace cpu;
#else
using thrust::optional;
template<typename T>
using vector = field<T>;

using cuda::std::apply;
using cuda::std::get;
using cuda::std::make_tuple;
using cuda::std::tuple;

using thrust::plus;

using thrust::copy;
using thrust::fill;
using thrust::fill_n;

namespace device
{
// no parallelization, because as a __device__ function these are called per thread

template<class InputIt, class T, class BinaryReductionOp, class UnaryTransformOp>
__device__ __forceinline__ T
transform_reduce( InputIt first, InputIt last, T init, BinaryReductionOp reduce, UnaryTransformOp transform )
{
    while( first != last )
        init = reduce( init, transform( *( first++ ) ) );

    return init;
};

template<class InputIt1, class InputIt2, class T, class BinaryReductionOp, class BinaryTransformOp>
__device__ __forceinline__ T transform_reduce(
    InputIt1 first1, InputIt1 last1, InputIt2 first2, T init, BinaryReductionOp reduce, BinaryTransformOp transform )
{
    while( first1 != last1 )
        init = reduce( init, transform( *( first1++ ), *( first2++ ) ) );

    return init;
};

template<class InputIt, class OutputIt, class UnaryOp>
__device__ __forceinline__ OutputIt transform( InputIt first, InputIt last, OutputIt d_first, UnaryOp unary_op )
{
    while( first != last )
        *( d_first++ ) = unary_op( *( first++ ) );

    return d_first;
};

template<class InputIt1, class InputIt2, class OutputIt, class BinaryOp>
__device__ __forceinline__ OutputIt
transform( InputIt1 first1, InputIt1 last1, InputIt2 first2, OutputIt d_first, BinaryOp binary_op )
{
    while( first1 != last1 )
        *( d_first++ ) = binary_op( *( first1++ ), *( first2++ ) );

    return d_first;
};

} // namespace device

namespace kernel
{

template<class InputIt, class OutputIt, class UnaryOp>
__global__ void transform_n( InputIt first, int N, OutputIt d_first, UnaryOp unary_op )
{
    static_assert( std::is_pointer<InputIt>::value );
    static_assert( std::is_pointer<OutputIt>::value );

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        d_first[idx] = unary_op( first[idx] );
    }
};

template<class InputIt1, class InputIt2, class OutputIt, class BinaryOp>
__global__ void transform_n( InputIt1 first1, int N, InputIt2 first2, OutputIt d_first, BinaryOp binary_op )
{
    static_assert( std::is_pointer<InputIt1>::value );
    static_assert( std::is_pointer<InputIt2>::value );
    static_assert( std::is_pointer<OutputIt>::value );

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        d_first[idx] = binary_op( first1[idx], first2[idx] );
    }
};

} // namespace kernel

namespace host
{

template<class InputIt, class T, class BinaryReductionOp, class UnaryTransformOp>
__host__ T transform_reduce( InputIt first, InputIt last, T init, BinaryReductionOp reduce, UnaryTransformOp transform )
{
    const int N = std::distance( first, last );

    field<T> ret( 1, init );

    auto t_first = cub::TransformInputIterator<T, UnaryTransformOp, InputIt>( first, transform );

    // Determine temporary storage size and allocate
    void * d_temp_storage     = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Reduce( d_temp_storage, temp_storage_bytes, t_first, ret.data(), N, reduce, init );
    cudaMalloc( &d_temp_storage, temp_storage_bytes );
    // Reduction
    cub::DeviceReduce::Reduce( d_temp_storage, temp_storage_bytes, t_first, ret.data(), N, reduce, init );
    CU_CHECK_AND_SYNC();
    cudaFree( d_temp_storage );
    return ret[0];
}

template<class InputIt, class OutputIt, class UnaryOp>
__host__ OutputIt transform( InputIt first, InputIt last, OutputIt d_first, UnaryOp unary_op )
{
    const int N                = std::distance( first, last );
    static const int blockSize = []()
    {
        int blockSize   = 0;
        int minGridSize = 0;
        CU_HANDLE_ERROR( cudaOccupancyMaxPotentialBlockSize(
            &blockSize, &minGridSize,
            &Backend::kernel::transform_n<
                std::decay_t<decltype( raw_pointer_cast( first ) )>,
                std::decay_t<decltype( raw_pointer_cast( d_first ) )>, UnaryOp> ) );
        return blockSize;
    }();
    Backend::kernel::transform_n<<<( N + blockSize - 1 ) / blockSize, blockSize>>>(
        raw_pointer_cast( first ), N, raw_pointer_cast( d_first ), unary_op );
    CU_CHECK_AND_SYNC();
    return std::next( d_first, N );
}

template<class InputIt1, class InputIt2, class OutputIt, class BinaryOp>
__host__ OutputIt transform( InputIt1 first1, InputIt1 last1, InputIt2 first2, OutputIt d_first, BinaryOp binary_op )
{
    const int N                = std::distance( first1, last1 );
    static const int blockSize = []()
    {
        int blockSize   = 0;
        int minGridSize = 0;
        CU_HANDLE_ERROR( cudaOccupancyMaxPotentialBlockSize(
            &blockSize, &minGridSize,
            &Backend::kernel::transform_n<
                std::decay_t<decltype( raw_pointer_cast( first1 ) )>,
                std::decay_t<decltype( raw_pointer_cast( first2 ) )>,
                std::decay_t<decltype( raw_pointer_cast( d_first ) )>, BinaryOp> ) );
        return blockSize;
    }();

    Backend::kernel::transform_n<<<( N + blockSize - 1 ) / blockSize, blockSize>>>(
        raw_pointer_cast( first1 ), N, raw_pointer_cast( first2 ), raw_pointer_cast( d_first ), binary_op );
    CU_CHECK_AND_SYNC();
    return std::next( d_first, N );
}

} // namespace host

// requires that `reduce` is a `__host__ __device__` functor
template<class InputIt, class T, class BinaryReductionOp, class UnaryTransformOp>
__host__ auto
transform_reduce( InputIt first, InputIt last, T init, BinaryReductionOp reduce, UnaryTransformOp transform ) ->
    typename std::enable_if<is_host_iterator<typename std::decay<InputIt>::type>::value, T>::type
{
    const int N = std::distance( first, last );

    field<T> ret( 1, init );

    using pointer_t = typename std::iterator_traits<InputIt>::pointer;

    auto t_first = cub::TransformInputIterator<T, UnaryTransformOp, pointer_t>( raw_pointer_cast( first ), transform );

    // Determine temporary storage size and allocate
    void * d_temp_storage     = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Reduce( d_temp_storage, temp_storage_bytes, t_first, ret.data(), N, reduce, init );
    cudaMalloc( &d_temp_storage, temp_storage_bytes );
    // Reduction
    cub::DeviceReduce::Reduce( d_temp_storage, temp_storage_bytes, t_first, ret.data(), N, reduce, init );
    CU_CHECK_AND_SYNC();
    cudaFree( d_temp_storage );
    return ret[0];
}

template<class InputIt, class T, class BinaryReductionOp, class UnaryTransformOp>
__host__ __device__ auto
transform_reduce( InputIt first, InputIt last, T init, BinaryReductionOp && reduce, UnaryTransformOp && transform ) ->
    typename std::enable_if<!is_host_iterator<typename std::decay<InputIt>::type>::value, T>::type
{
    return
#ifdef __CUDA_ARCH__
        Backend::device
#else
        Backend::host
#endif
        ::transform_reduce(
            first, last, init, std::forward<BinaryReductionOp>( reduce ), std::forward<UnaryTransformOp>( transform ) );
};

template<class InputIt1, class InputIt2, class T, class BinaryReductionOp, class BinaryTransformOp>
__host__ __device__ T transform_reduce(
    InputIt1 first1, InputIt1 last1, InputIt2 first2, T && init, BinaryReductionOp && reduce,
    BinaryTransformOp && transform )
{
    return Backend::transform_reduce(
        thrust::make_zip_iterator( first1, first2 ),
        thrust::make_zip_iterator( last1, first2 + std::distance( first1, last1 ) ), std::forward<T>( init ),
        std::forward<BinaryReductionOp>( reduce ),
        thrust::make_zip_function( std::forward<BinaryTransformOp>( transform ) ) );
};

template<class InputIt, class OutputIt, class UnaryOp>
__host__ auto transform( InputIt first, InputIt last, OutputIt && d_first, UnaryOp && unary_op ) ->
    typename std::enable_if<is_host_iterator<typename std::decay<InputIt>::type>::value, OutputIt>::type
{
    return Backend::host::transform(
        first, last, std::forward<OutputIt>( d_first ), std::forward<UnaryOp>( unary_op ) );
}

template<class InputIt, class OutputIt, class UnaryOp>
__host__ __device__ auto transform( InputIt first, InputIt last, OutputIt && d_first, UnaryOp && unary_op ) ->
    typename std::enable_if<!is_host_iterator<typename std::decay<InputIt>::type>::value, OutputIt>::type
{
    return
#ifdef __CUDA_ARCH__
        Backend::device
#else
        Backend::host
#endif
        ::transform( first, last, std::forward<OutputIt>( d_first ), std::forward<UnaryOp>( unary_op ) );
}

template<class InputIt1, class InputIt2, class OutputIt, class BinaryOp>
__host__ auto
transform( InputIt1 first1, InputIt1 last1, InputIt2 && first2, OutputIt && d_first, BinaryOp && binary_op ) ->
    typename std::enable_if<
        std::conjunction<
            is_host_iterator<typename std::decay<InputIt1>::type>,
            is_host_iterator<typename std::decay<InputIt2>::type>>::value,
        OutputIt>::type
{
    return Backend::host::transform(
        first1, last1, std::forward<InputIt2>( first2 ), std::forward<OutputIt>( d_first ),
        std::forward<BinaryOp>( binary_op ) );
}

template<class InputIt1, class InputIt2, class OutputIt, class BinaryOp>
__host__ __device__ auto
transform( InputIt1 first1, InputIt1 last1, InputIt2 && first2, OutputIt && d_first, BinaryOp && binary_op ) ->
    typename std::enable_if<
        !std::conjunction<
            is_host_iterator<typename std::decay<InputIt1>::type>,
            is_host_iterator<typename std::decay<InputIt2>::type>>::value,
        OutputIt>::type
{
    return
#ifdef __CUDA_ARCH__
        Backend::device
#else
        Backend::host
#endif
        ::transform(
            first1, last1, std::forward<InputIt2>( first2 ), std::forward<OutputIt>( d_first ),
            std::forward<BinaryOp>( binary_op ) );
}

#endif

// TODO: migrate all of these to the new stdlib conforming backend
// This still needs an implementation of a `counting_iterator` class that should be wrapped around to index based bounds in the CUDA implementation,
// because these are closer to the CUDA kernel implementation
namespace par
{

#ifdef SPIRIT_USE_CUDA

template<typename F>
__global__ void cu_apply( int N, F f )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
        f( idx );
}

// vf1[i] = f( vf2[i] )
template<typename F>
void apply( int N, F f )
{
    cu_apply<<<( N + 1023 ) / 1024, 1024>>>( N, f );
    CU_CHECK_AND_SYNC();
}

template<typename F>
scalar reduce( int N, const F f )
{
    static scalarfield sf( N, 0 );
    // Vectormath::fill(sf, 0);

    if( sf.size() != N )
        sf.resize( N );

    auto s = sf.data();
    apply( N, [f, s] SPIRIT_LAMBDA( int idx ) { s[idx] = f( idx ); } );

    static scalarfield ret( 1, 0 );

    // Determine temporary storage size and allocate
    void * d_temp_storage     = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum( d_temp_storage, temp_storage_bytes, sf.data(), ret.data(), sf.size() );
    cudaMalloc( &d_temp_storage, temp_storage_bytes );
    // Reduction
    cub::DeviceReduce::Sum( d_temp_storage, temp_storage_bytes, sf.data(), ret.data(), sf.size() );
    CU_CHECK_AND_SYNC();
    cudaFree( d_temp_storage );
    return ret[0];
}

template<typename A, typename F>
scalar reduce( const field<A> & vf1, const F f )
{
    // TODO: remove the reliance on a temporary scalar field (maybe thrust::dot with generalized operations)
    // We also use this workaround for a single field as argument, because cub does not support non-commutative
    // reduction operations

    int n = vf1.size();
    static scalarfield sf( n, 0 );
    // Vectormath::fill(sf, 0);

    if( sf.size() != vf1.size() )
        sf.resize( vf1.size() );

    auto s  = sf.data();
    auto v1 = vf1.data();
    apply( n, [f, s, v1] SPIRIT_LAMBDA( int idx ) { s[idx] = f( v1[idx] ); } );

    static scalarfield ret( 1, 0 );
    // Vectormath::fill(ret, 0);

    // Determine temporary storage size and allocate
    void * d_temp_storage     = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum( d_temp_storage, temp_storage_bytes, sf.data(), ret.data(), sf.size() );
    cudaMalloc( &d_temp_storage, temp_storage_bytes );
    // Reduction
    cub::DeviceReduce::Sum( d_temp_storage, temp_storage_bytes, sf.data(), ret.data(), sf.size() );
    CU_CHECK_AND_SYNC();
    cudaFree( d_temp_storage );
    return ret[0];
}

template<typename A, typename B, typename F>
scalar reduce( const field<A> & vf1, const field<B> & vf2, const F f )
{
    // TODO: remove the reliance on a temporary scalar field (maybe thrust::dot with generalized operations)
    int n = vf1.size();
    static scalarfield sf( n, 0 );
    // Vectormath::fill(sf, 0);

    if( sf.size() != vf1.size() )
        sf.resize( vf1.size() );

    auto s  = sf.data();
    auto v1 = vf1.data();
    auto v2 = vf2.data();
    apply( n, [f, s, v1, v2] SPIRIT_LAMBDA( int idx ) { s[idx] = f( v1[idx], v2[idx] ); } );

    static scalarfield ret( 1, 0 );
    // Vectormath::fill(ret, 0);
    // Determine temporary storage size and allocate
    void * d_temp_storage     = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum( d_temp_storage, temp_storage_bytes, sf.data(), ret.data(), sf.size() );
    cudaMalloc( &d_temp_storage, temp_storage_bytes );
    // Reduction
    cub::DeviceReduce::Sum( d_temp_storage, temp_storage_bytes, sf.data(), ret.data(), sf.size() );
    CU_CHECK_AND_SYNC();
    cudaFree( d_temp_storage );
    return ret[0];
}

// vf1[i] = f( vf2[i] )
template<typename A, typename B, typename F>
__global__ void cu_set_lambda( A * vf1, const B * vf2, F f, int N )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        vf1[idx] = f( vf2[idx] );
    }
}

// vf1[i] = f( vf2[i] )
template<typename A, typename B, typename F>
void set( field<A> & vf1, const field<B> & vf2, F f )
{
    int N = vf1.size();
    cu_set_lambda<<<( N + 1023 ) / 1024, 1024>>>( vf1.data(), vf2.data(), f, N );
    CU_CHECK_AND_SYNC();
}

#else

template<typename F>
scalar reduce( int N, const F f )
{
    scalar res = 0;
#pragma omp parallel for reduction( + : res )
    for( unsigned int idx = 0; idx < N; ++idx )
    {
        res += f( idx );
    }
    return res;
}

template<typename A, typename F>
scalar reduce( const field<A> & vf1, const F f )
{
    scalar res = 0;
#pragma omp parallel for reduction( + : res )
    for( unsigned int idx = 0; idx < vf1.size(); ++idx )
    {
        res += f( vf1[idx] );
    }
    return res;
}

// result = sum_i  f( vf1[i], vf2[i] )
template<typename A, typename B, typename F>
scalar reduce( const field<A> & vf1, const field<B> & vf2, const F & f )
{
    scalar res = 0;
#pragma omp parallel for reduction( + : res )
    for( unsigned int idx = 0; idx < vf1.size(); ++idx )
    {
        res += f( vf1[idx], vf2[idx] );
    }
    return res;
}

// vf1[i] = f( vf2[i] )
template<typename A, typename B, typename F>
void set( field<A> & vf1, const field<B> & vf2, const F & f )
{
#pragma omp parallel for
    for( unsigned int idx = 0; idx < vf1.size(); ++idx )
    {
        vf1[idx] = f( vf2[idx] );
    }
}

// f( vf1[idx], idx ) for all i
template<typename F>
void apply( int N, const F & f )
{
#pragma omp parallel for
    for( unsigned int idx = 0; idx < N; ++idx )
    {
        f( idx );
    }
}

#endif
} // namespace par

} // namespace Backend

} // namespace Engine
#endif
