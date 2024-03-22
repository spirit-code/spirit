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
#else
#define SPIRIT_CPU_PAR
#endif

// clang-format off
#ifdef SPIRIT_USE_CUDA
    #include <thrust/copy.h>
    #include <thrust/fill.h>
    #include <thrust/for_each.h>
    #include <thrust/optional.h>
    #include <thrust/transform.h>
    #include <thrust/transform_reduce.h>
    #include <thrust/universal_vector.h>
    #include <thrust/iterator/zip_iterator.h>
    #include <thrust/zip_function.h>

    #include <thrust/execution_policy.h>

    #define THRUST_IGNORE_CUB_VERSION_CHECK
    #include <cub/cub.cuh>
    #include <cub/iterator/transform_input_iterator.cuh>

    #define SPIRIT_PAR

    #define SPIRIT_LAMBDA __device__
#else
    #define SPIRIT_PAR SPIRIT_CPU_PAR

    #define SPIRIT_LAMBDA
#endif
// clang-format on

namespace Engine
{

namespace Backend
{

namespace cpu
{

using std::optional;
using std::vector;

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
using vector = thrust::universal_vector<T>;

using thrust::plus;

using thrust::copy;
using thrust::fill;
using thrust::fill_n;
using thrust::for_each;
using thrust::transform;

namespace device
{

template<class InputIt, class T, class BinaryReductionOp, class UnaryTransformOp>
__device__ __forceinline__ T
transform_reduce( InputIt first, InputIt last, T init, BinaryReductionOp reduce, UnaryTransformOp transform )
{
    {
        auto t_it   = cub::TransformInputIterator<T, UnaryTransformOp, InputIt>( first, transform );
        auto t_last = cub::TransformInputIterator<T, UnaryTransformOp, InputIt>( last, transform );
        // no parallelization, because as a __device__ function this is called per thread
        for( ; t_it != t_last; ++t_it )
        {
            init = reduce( init, *t_it );
        }
    }
    return init;
}

} // namespace device

// requires that `reduce` is a `__host__ __device__` functor
template<class InputIt, class T, class BinaryReductionOp, class UnaryTransformOp>
__host__ __device__ T
transform_reduce( InputIt first, InputIt last, T && init, BinaryReductionOp && reduce, UnaryTransformOp && transform )
{
#ifdef __CUDA_ARCH__
    return Backend::device::transform_reduce(
        first, last, std::forward<T>( init ), std::forward<BinaryReductionOp>( reduce ),
        std::forward<UnaryTransformOp>( transform ) );
#else
    return thrust::transform_reduce(
        first, last, std::forward<UnaryTransformOp>( transform ), std::forward<T>( init ),
        std::forward<BinaryReductionOp>( reduce ) );
#endif
};

template<class DerivedPolicy, class InputIt, class T, class BinaryReductionOp, class UnaryTransformOp>
__host__ __device__ T transform_reduce(
    const thrust::detail::execution_policy_base<DerivedPolicy> & policy, InputIt first1, InputIt last1, T && init,
    BinaryReductionOp && reduce, UnaryTransformOp && transform )
{
    return thrust::transform_reduce(
        policy, first1, last1, std::forward<UnaryTransformOp>( transform ), std::forward<T>( init ),
        std::forward<BinaryReductionOp>( reduce ) );
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

template<class DerivedPolicy, class InputIt1, class InputIt2, class T, class BinaryReductionOp, class BinaryTransformOp>
__host__ __device__ T transform_reduce(
    const thrust::detail::execution_policy_base<DerivedPolicy> & policy, InputIt1 first1, InputIt1 last1,
    InputIt2 first2, T && init, BinaryReductionOp && reduce, BinaryTransformOp && transform )
{
    return thrust::transform_reduce(
        policy, thrust::make_zip_iterator( first1, first2 ),
        thrust::make_zip_iterator( last1, first2 + std::distance( first1, last1 ) ),
        thrust::make_zip_function( std::forward<BinaryTransformOp>( transform ) ), std::forward<T>( init ),
        std::forward<BinaryReductionOp>( reduce ) );
};
#endif

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
    void * d_temp_storage     = NULL;
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
    void * d_temp_storage     = NULL;
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
    void * d_temp_storage     = NULL;
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
