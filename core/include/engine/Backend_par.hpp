#pragma once
#ifndef SPIRIT_CORE_ENGINE_BACKEND_PAR_HPP
#define SPIRIT_CORE_ENGINE_BACKEND_PAR_HPP

#include <engine/Vectormath_Defines.hpp>

// clang-format off
#ifdef SPIRIT_USE_CUDA
    #include <cub/cub.cuh>
    #define SPIRIT_LAMBDA __device__
#else
    #define SPIRIT_LAMBDA
#endif
// clang-format on

namespace Engine
{
namespace Backend
{
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