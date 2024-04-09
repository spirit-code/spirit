#ifdef SPIRIT_USE_CUDA

#include <engine/Backend.hpp>
#include <engine/Vectormath.hpp>
#include <utility/Constants.hpp>
#include <utility/Exception.hpp>
#include <utility/Logging.hpp>

#include <Eigen/Dense>

#include <stdio.h>
#include <algorithm>
#include <iostream>

#include <curand.h>
#include <curand_kernel.h>

#include <cub/cub.cuh>

using namespace Utility;
using Utility::Constants::Pi;

// CUDA Version
namespace Engine
{
namespace Vectormath
{

void get_random_vector( std::uniform_real_distribution<scalar> & distribution, std::mt19937 & prng, Vector3 & vec )
{
    for( int dim = 0; dim < 3; ++dim )
    {
        vec[dim] = distribution( prng );
    }
}

// TODO: improve random number generation - this one might give undefined behaviour!
__global__ void cu_get_random_vectorfield( Vector3 * xi, const size_t N )
{
    unsigned long long subsequence = 0;
    unsigned long long offset      = 0;

    curandState_t state;
    for( int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x )
    {
        curand_init( idx, subsequence, offset, &state );
        for( int dim = 0; dim < 3; ++dim )
        {
            xi[idx][dim] = llroundf( curand_uniform( &state ) ) * 2 - 1;
        }
    }
}
void get_random_vectorfield(
    std::uniform_real_distribution<scalar> & distribution, std::mt19937 & prng, vectorfield & xi )
{
    unsigned int n = xi.size();
    cu_get_random_vectorfield<<<( n + 1023 ) / 1024, 1024>>>( raw_pointer_cast( xi.data() ), n );
    CU_CHECK_AND_SYNC();
}

void get_random_vector_unitsphere(
    std::uniform_real_distribution<scalar> & distribution, std::mt19937 & prng, Vector3 & vec )
{
    scalar v_z = distribution( prng );
    scalar phi = distribution( prng );

    scalar r_xy = std::sqrt( 1 - v_z * v_z );

    vec[0] = r_xy * std::cos( 2 * Pi * phi );
    vec[1] = r_xy * std::sin( 2 * Pi * phi );
    vec[2] = v_z;
}
// __global__ void cu_get_random_vectorfield_unitsphere(Vector3 * xi, const size_t N)
// {
//     unsigned long long subsequence = 0;
//     unsigned long long offset= 0;

//     curandState_t state;
//     for(int idx = blockIdx.x * blockDim.x + threadIdx.x;
//         idx < N;
//         idx +=  blockDim.x * gridDim.x)
//     {
//         curand_init(idx,subsequence,offset,&state);

//         scalar v_z = llroundf(curand_uniform(&state))*2-1;
//         scalar phi = llroundf(curand_uniform(&state))*2-1;

// 	    scalar r_xy = std::sqrt(1 - v_z*v_z);

//         xi[idx][0] = r_xy * std::cos(2*Pi*phi);
//         xi[idx][1] = r_xy * std::sin(2*Pi*phi);
//         xi[idx][2] = v_z;
//     }
// }
// void get_random_vectorfield_unitsphere(std::mt19937 & prng, vectorfield & xi)
// {
//     int n = xi.size();
//     cu_get_random_vectorfield<<<(n+1023)/1024, raw_pointer_cast( 1024>>>(xi.data() ), n);
//     CU_CHECK_AND_SYNC();
// }
// The above CUDA implementation does not work correctly.
void get_random_vectorfield_unitsphere( std::mt19937 & prng, vectorfield & xi )
{
    // PRNG gives RN [-1,1] -> multiply with epsilon
    auto distribution = std::uniform_real_distribution<scalar>( -1, 1 );
// TODO: parallelization of this is actually not quite so trivial
#pragma omp parallel for
    for( unsigned int i = 0; i < xi.size(); ++i )
    {
        get_random_vector_unitsphere( distribution, prng, xi[i] );
    }
}

/////////////////////////////////////////////////////////////////

__global__ void cu_fill( scalar * sf, scalar s, const size_t N )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        sf[idx] = s;
    }
}
void fill( scalarfield & sf, scalar s )
{
    unsigned int n = sf.size();
    cu_fill<<<( n + 1023 ) / 1024, 1024>>>( raw_pointer_cast( sf.data() ), s, n );
    CU_CHECK_AND_SYNC();
}
__global__ void cu_fill_mask( scalar * sf, scalar s, const int * mask, const size_t N )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        sf[idx] = mask[idx] * s;
    }
}
void fill( scalarfield & sf, scalar s, const intfield & mask )
{
    unsigned int n = sf.size();
    cu_fill_mask<<<( n + 1023 ) / 1024, 1024>>>( raw_pointer_cast( sf.data() ), s, raw_pointer_cast( mask.data() ), n );
    CU_CHECK_AND_SYNC();
}

__global__ void cu_scale( scalar * sf, scalar s, const size_t N )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        sf[idx] *= s;
    }
}
void scale( scalarfield & sf, scalar s )
{
    unsigned int n = sf.size();
    cu_scale<<<( n + 1023 ) / 1024, 1024>>>( raw_pointer_cast( sf.data() ), s, n );
    CU_CHECK_AND_SYNC();
}

__global__ void cu_add( scalar * sf, scalar s, const size_t N )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        sf[idx] += s;
    }
}
void add( scalarfield & sf, scalar s )
{
    unsigned int n = sf.size();
    cu_add<<<( n + 1023 ) / 1024, 1024>>>( raw_pointer_cast( sf.data() ), s, n );
    cudaDeviceSynchronize();
}

scalar sum( const scalarfield & sf )
{
    static scalarfield ret( 1, 0 );
    Vectormath::fill( ret, 0 );
    // Determine temporary storage size and allocate
    void * d_temp_storage     = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(
        d_temp_storage, temp_storage_bytes, raw_pointer_cast( sf.data() ), raw_pointer_cast( ret.data() ), sf.size() );
    cudaMalloc( &d_temp_storage, temp_storage_bytes );
    // Reduction
    cub::DeviceReduce::Sum(
        d_temp_storage, temp_storage_bytes, raw_pointer_cast( sf.data() ), raw_pointer_cast( ret.data() ), sf.size() );
    CU_CHECK_AND_SYNC();
    cudaFree( d_temp_storage );
    return ret[0];
}

scalar mean( const scalarfield & sf )
{
    return sum( sf ) / sf.size();
}

__global__ void cu_divide( const scalar * numerator, const scalar * denominator, scalar * out, const size_t N )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        out[idx] += numerator[idx] / denominator[idx];
    }
}
void divide( const scalarfield & numerator, const scalarfield & denominator, scalarfield & out )
{
    unsigned int n = numerator.size();
    cu_divide<<<( n + 1023 ) / 1024, 1024>>>(
        raw_pointer_cast( numerator.data() ), raw_pointer_cast( denominator.data() ), raw_pointer_cast( out.data() ),
        n );
    CU_CHECK_AND_SYNC();
}

void set_range( scalarfield & sf, scalar sf_min, scalar sf_max )
{
#pragma omp parallel for
    for( unsigned int i = 0; i < sf.size(); ++i )
        sf[i] = std::min( std::max( sf_min, sf[i] ), sf_max );
}

__global__ void cu_fill( Vector3 * vf1, Vector3 v2, const size_t N )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        vf1[idx] = v2;
    }
}
void fill( vectorfield & vf, const Vector3 & v )
{
    unsigned int n = vf.size();
    cu_fill<<<( n + 1023 ) / 1024, 1024>>>( raw_pointer_cast( vf.data() ), v, n );
    CU_CHECK_AND_SYNC();
}
__global__ void cu_fill_mask( Vector3 * vf1, Vector3 v2, const int * mask, const size_t N )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        vf1[idx] = v2;
    }
}
void fill( vectorfield & vf, const Vector3 & v, const intfield & mask )
{
    unsigned int n = vf.size();
    cu_fill_mask<<<( n + 1023 ) / 1024, 1024>>>( raw_pointer_cast( vf.data() ), v, raw_pointer_cast( mask.data() ), n );
    CU_CHECK_AND_SYNC();
}

__global__ void cu_normalize_vectors( Vector3 * vf, const size_t N )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        vf[idx].normalize();
    }
}
void normalize_vectors( vectorfield & vf )
{
    unsigned int n = vf.size();
    cu_normalize_vectors<<<( n + 1023 ) / 1024, 1024>>>( raw_pointer_cast( vf.data() ), n );
    CU_CHECK_AND_SYNC();
}

__global__ void cu_norm( const Vector3 * vf, scalar * norm, const size_t N )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        norm[idx] = vf[idx].norm();
    }
}
void norm( const vectorfield & vf, scalarfield & norm )
{
    unsigned int n = vf.size();
    cu_norm<<<( n + 1023 ) / 1024, 1024>>>( raw_pointer_cast( vf.data() ), raw_pointer_cast( norm.data() ), n );
    CU_CHECK_AND_SYNC();
}

// Functor for finding the maximum absolute value
// struct CustomMaxAbs
// {
//     template <typename T>
//     __device__ __forceinline__
//     T operator()(const T &a, const T &b) const {
//         return (a > b) ? a : b;
//     }
// };
scalar max_abs_component( const vectorfield & vf )
{
    // Declare, allocate, and initialize device-accessible pointers for input and output
    // CustomMaxAbs    max_op;
    size_t N = 3 * vf.size();
    scalarfield out( 1, 0 );
    scalar init = 0;
    // Determine temporary device storage requirements
    void * d_temp_storage     = NULL;
    size_t temp_storage_bytes = 0;
    auto lam = [] __host__ __device__( const scalar & a, const scalar & b ) { return ( a > b ) ? a : b; };
    cub::DeviceReduce::Reduce(
        d_temp_storage, temp_storage_bytes, raw_pointer_cast( vf[0].data() ), raw_pointer_cast( out.data() ), N, lam,
        init );
    // Allocate temporary storage
    cudaMalloc( &d_temp_storage, temp_storage_bytes );
    // Run reduction
    cub::DeviceReduce::Reduce(
        d_temp_storage, temp_storage_bytes, raw_pointer_cast( vf[0].data() ), raw_pointer_cast( out.data() ), N, lam,
        init );
    CU_CHECK_AND_SYNC();
    cudaFree( d_temp_storage );
    return std::abs( out[0] );
}

scalar max_norm( const vectorfield & vf )
{
    static scalarfield ret( 1, 0 );

    // Declare, allocate, and initialize device-accessible pointers for input and output
    size_t N = vf.size();
    scalarfield temp( N, 0 );
    auto o = raw_pointer_cast( temp.data() );
    auto v = raw_pointer_cast( vf.data() );
    Backend::for_each_n(
        SPIRIT_PAR Backend::make_counting_iterator( 0 ), N,
        [o, v] SPIRIT_LAMBDA( int idx )
        { o[idx] = v[idx][0] * v[idx][0] + v[idx][1] * v[idx][1] + v[idx][2] * v[idx][2]; } );

    void * d_temp_storage     = NULL;
    size_t temp_storage_bytes = 0;
    auto lam = [] __host__ __device__( const scalar & a, const scalar & b ) { return ( a > b ) ? a : b; };

    scalar init = 0;
    cub::DeviceReduce::Reduce(
        d_temp_storage, temp_storage_bytes, raw_pointer_cast( temp.data() ), raw_pointer_cast( ret.data() ), N, lam,
        init );
    // Allocate temporary storage
    cudaMalloc( &d_temp_storage, temp_storage_bytes );
    // Run reduction
    cub::DeviceReduce::Reduce(
        d_temp_storage, temp_storage_bytes, raw_pointer_cast( temp.data() ), raw_pointer_cast( ret.data() ), N, lam,
        init );
    CU_CHECK_AND_SYNC();
    cudaFree( d_temp_storage );
    return std::sqrt( ret[0] );
}

__global__ void cu_scale( Vector3 * vf1, scalar sc, const size_t N )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        vf1[idx] *= sc;
    }
}
void scale( vectorfield & vf, const scalar & sc )
{
    unsigned int n = vf.size();
    cu_scale<<<( n + 1023 ) / 1024, 1024>>>( raw_pointer_cast( vf.data() ), sc, n );
    CU_CHECK_AND_SYNC();
}

__global__ void cu_scale( Vector3 * vf1, const scalar * sf, bool inverse, const size_t N )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        if( inverse )
            vf1[idx] /= sf[idx];
        else
            vf1[idx] *= sf[idx];
    }
}
void scale( vectorfield & vf, const scalarfield & sf, bool inverse )
{
    unsigned int n = vf.size();
    cu_scale<<<( n + 1023 ) / 1024, 1024>>>( raw_pointer_cast( vf.data() ), raw_pointer_cast( sf.data() ), inverse, n );
    CU_CHECK_AND_SYNC();
}

// Functor for adding Vector3's
struct CustomAdd
{
    template<typename T>
    __device__ __forceinline__ T operator()( const T & a, const T & b ) const
    {
        return a + b;
    }
};
Vector3 sum( const vectorfield & vf )
{
    static vectorfield ret( 1, { 0, 0, 0 } );
    Vectormath::fill( ret, { 0, 0, 0 } );
    // Declare, allocate, and initialize device-accessible pointers for input and output
    CustomAdd add_op;
    static const Vector3 init{ 0, 0, 0 };
    // Determine temporary device storage requirements
    void * d_temp_storage     = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Reduce(
        d_temp_storage, temp_storage_bytes, raw_pointer_cast( vf.data() ), raw_pointer_cast( ret.data() ), vf.size(),
        add_op, init );
    // Allocate temporary storage
    cudaMalloc( &d_temp_storage, temp_storage_bytes );
    // Run reduction
    cub::DeviceReduce::Reduce(
        d_temp_storage, temp_storage_bytes, raw_pointer_cast( vf.data() ), raw_pointer_cast( ret.data() ), vf.size(),
        add_op, init );
    CU_CHECK_AND_SYNC();
    cudaFree( d_temp_storage );
    return ret[0];
}

Vector3 mean( const vectorfield & vf )
{
    return sum( vf ) / vf.size();
}

__global__ void cu_dot( const Vector3 * vf1, const Vector3 * vf2, scalar * out, const size_t N )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        out[idx] = vf1[idx].dot( vf2[idx] );
    }
}

scalar dot( const vectorfield & vf1, const vectorfield & vf2 )
{
    unsigned int n = vf1.size();
    static scalarfield sf( n, 0 );

    if( sf.size() != vf1.size() )
        sf.resize( vf1.size() );

    Vectormath::fill( sf, 0 );
    scalar ret;

    // Dot product
    cu_dot<<<( n + 1023 ) / 1024, 1024>>>(
        raw_pointer_cast( vf1.data() ), raw_pointer_cast( vf2.data() ), raw_pointer_cast( sf.data() ), n );
    CU_CHECK_AND_SYNC();

    // reduction
    ret = sum( sf );
    return ret;
}

void dot( const vectorfield & vf1, const vectorfield & vf2, scalarfield & s )
{
    unsigned int n = vf1.size();

    // Dot product
    cu_dot<<<( n + 1023 ) / 1024, 1024>>>(
        raw_pointer_cast( vf1.data() ), raw_pointer_cast( vf2.data() ), raw_pointer_cast( s.data() ), n );
    CU_CHECK_AND_SYNC();
}

__global__ void cu_scalardot( const scalar * s1, const scalar * s2, scalar * out, const size_t N )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        out[idx] = s1[idx] * s2[idx];
    }
}
// computes the product of scalars in s1 and s2
// s1 and s2 are scalarfields
void dot( const scalarfield & s1, const scalarfield & s2, scalarfield & out )
{
    unsigned int n = s1.size();

    // Dot product
    cu_scalardot<<<( n + 1023 ) / 1024, 1024>>>(
        raw_pointer_cast( s1.data() ), raw_pointer_cast( s2.data() ), raw_pointer_cast( out.data() ), n );
    CU_CHECK_AND_SYNC();
}

__global__ void cu_cross( const Vector3 * vf1, const Vector3 * vf2, Vector3 * out, const size_t N )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        out[idx] = vf1[idx].cross( vf2[idx] );
    }
}
// The wrapper for the calling of the actual kernel
void cross( const vectorfield & vf1, const vectorfield & vf2, vectorfield & s )
{
    unsigned int n = vf1.size();

    // Dot product
    cu_cross<<<( n + 1023 ) / 1024, 1024>>>(
        raw_pointer_cast( vf1.data() ), raw_pointer_cast( vf2.data() ), raw_pointer_cast( s.data() ), n );
    CU_CHECK_AND_SYNC();
}

__global__ void cu_add_c_a( const scalar c, Vector3 a, Vector3 * out, const size_t N )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        out[idx] += c * a;
    }
}
// out[i] += c*a
void add_c_a( const scalar & c, const Vector3 & a, vectorfield & out )
{
    unsigned int n = out.size();
    cu_add_c_a<<<( n + 1023 ) / 1024, 1024>>>( c, a, raw_pointer_cast( out.data() ), n );
    CU_CHECK_AND_SYNC();
}

__global__ void cu_add_c_a2( const scalar c, const Vector3 * a, Vector3 * out, const size_t N )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        out[idx] += c * a[idx];
    }
}
// out[i] += c*a[i]
void add_c_a( const scalar & c, const vectorfield & a, vectorfield & out )
{
    unsigned int n = out.size();
    cu_add_c_a2<<<( n + 1023 ) / 1024, 1024>>>( c, raw_pointer_cast( a.data() ), raw_pointer_cast( out.data() ), n );
    CU_CHECK_AND_SYNC();
}

__global__ void cu_add_c_a2_mask( const scalar c, const Vector3 * a, Vector3 * out, const int * mask, const size_t N )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        out[idx] += c * mask[idx] * a[idx];
    }
}
// out[i] += c*a[i]
void add_c_a( const scalar & c, const vectorfield & a, vectorfield & out, const intfield & mask )
{
    unsigned int n = out.size();
    cu_add_c_a2_mask<<<( n + 1023 ) / 1024, 1024>>>(
        c, raw_pointer_cast( a.data() ), raw_pointer_cast( out.data() ), raw_pointer_cast( mask.data() ), n );
    CU_CHECK_AND_SYNC();
}

__global__ void cu_add_c_a3( const scalar * c, const Vector3 * a, Vector3 * out, const size_t N )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        out[idx] += c[idx] * a[idx];
    }
}
// out[i] += c[i]*a[i]
void add_c_a( const scalarfield & c, const vectorfield & a, vectorfield & out )
{
    unsigned int n = out.size();
    cu_add_c_a3<<<( n + 1023 ) / 1024, 1024>>>(
        raw_pointer_cast( c.data() ), raw_pointer_cast( a.data() ), raw_pointer_cast( out.data() ), n );
    CU_CHECK_AND_SYNC();
}

__global__ void cu_set_c_a( const scalar c, Vector3 a, Vector3 * out, const size_t N )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        out[idx] = c * a;
    }
}
// out[i] = c*a
void set_c_a( const scalar & c, const Vector3 & a, vectorfield & out )
{
    unsigned int n = out.size();
    cu_set_c_a<<<( n + 1023 ) / 1024, 1024>>>( c, a, raw_pointer_cast( out.data() ), n );
    CU_CHECK_AND_SYNC();
}
__global__ void cu_set_c_a_mask( const scalar c, Vector3 a, Vector3 * out, const int * mask, const size_t N )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        out[idx] = mask[idx] * c * a;
    }
}
// out[i] = c*a
void set_c_a( const scalar & c, const Vector3 & a, vectorfield & out, const intfield & mask )
{
    unsigned int n = out.size();
    cu_set_c_a_mask<<<( n + 1023 ) / 1024, 1024>>>(
        c, a, raw_pointer_cast( out.data() ), raw_pointer_cast( mask.data() ), n );
    CU_CHECK_AND_SYNC();
}

__global__ void cu_set_c_a2( const scalar c, const Vector3 * a, Vector3 * out, const size_t N )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        out[idx] = c * a[idx];
    }
}
// out[i] = c*a[i]
void set_c_a( const scalar & c, const vectorfield & a, vectorfield & out )
{
    unsigned int n = out.size();
    cu_set_c_a2<<<( n + 1023 ) / 1024, 1024>>>( c, raw_pointer_cast( a.data() ), raw_pointer_cast( out.data() ), n );
    CU_CHECK_AND_SYNC();
}
__global__ void cu_set_c_a2_mask( const scalar c, const Vector3 * a, Vector3 * out, const int * mask, const size_t N )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        out[idx] = mask[idx] * c * a[idx];
    }
}
// out[i] = c*a[i]
void set_c_a( const scalar & c, const vectorfield & a, vectorfield & out, const intfield & mask )
{
    unsigned int n = out.size();
    cu_set_c_a2_mask<<<( n + 1023 ) / 1024, 1024>>>(
        c, raw_pointer_cast( a.data() ), raw_pointer_cast( out.data() ), raw_pointer_cast( mask.data() ), n );
    CU_CHECK_AND_SYNC();
}

__global__ void cu_set_c_a3( const scalar * c, const Vector3 * a, Vector3 * out, const size_t N )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        out[idx] = c[idx] * a[idx];
    }
}
// out[i] = c[i]*a[i]
void set_c_a( const scalarfield & c, const vectorfield & a, vectorfield & out )
{
    unsigned int n = out.size();
    cu_set_c_a3<<<( n + 1023 ) / 1024, 1024>>>(
        raw_pointer_cast( c.data() ), raw_pointer_cast( a.data() ), raw_pointer_cast( out.data() ), n );
    CU_CHECK_AND_SYNC();
}

__global__ void cu_add_c_dot( const scalar c, Vector3 a, const Vector3 * b, scalar * out, const size_t N )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        out[idx] += c * a.dot( b[idx] );
    }
}
// out[i] += c * a*b[i]
void add_c_dot( const scalar & c, const Vector3 & a, const vectorfield & b, scalarfield & out )
{
    unsigned int n = out.size();
    cu_add_c_dot<<<( n + 1023 ) / 1024, 1024>>>(
        c, a, raw_pointer_cast( b.data() ), raw_pointer_cast( out.data() ), n );
    CU_CHECK_AND_SYNC();
}

__global__ void cu_add_c_dot( const scalar c, const Vector3 * a, const Vector3 * b, scalar * out, const size_t N )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        out[idx] += c * a[idx].dot( b[idx] );
    }
}
// out[i] += c * a[i]*b[i]
void add_c_dot( const scalar & c, const vectorfield & a, const vectorfield & b, scalarfield & out )
{
    unsigned int n = out.size();
    cu_add_c_dot<<<( n + 1023 ) / 1024, 1024>>>(
        c, raw_pointer_cast( a.data() ), raw_pointer_cast( b.data() ), raw_pointer_cast( out.data() ), n );
    CU_CHECK_AND_SYNC();
}

__global__ void cu_set_c_dot( const scalar c, Vector3 a, const Vector3 * b, scalar * out, const size_t N )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        out[idx] = c * a.dot( b[idx] );
    }
}
// out[i] = c * a*b[i]
void set_c_dot( const scalar & c, const Vector3 & a, const vectorfield & b, scalarfield & out )
{
    unsigned int n = out.size();
    cu_set_c_dot<<<( n + 1023 ) / 1024, 1024>>>(
        c, a, raw_pointer_cast( b.data() ), raw_pointer_cast( out.data() ), n );
    CU_CHECK_AND_SYNC();
}

__global__ void cu_set_c_dot( const scalar c, const Vector3 * a, const Vector3 * b, scalar * out, const size_t N )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        out[idx] = c * a[idx].dot( b[idx] );
    }
}
// out[i] = c * a[i]*b[i]
void set_c_dot( const scalar & c, const vectorfield & a, const vectorfield & b, scalarfield & out )
{
    unsigned int n = out.size();
    cu_set_c_dot<<<( n + 1023 ) / 1024, 1024>>>(
        c, raw_pointer_cast( a.data() ), raw_pointer_cast( b.data() ), raw_pointer_cast( out.data() ), n );
    CU_CHECK_AND_SYNC();
}

// out[i] += c * a x b[i]
__global__ void cu_add_c_cross( const scalar c, const Vector3 a, const Vector3 * b, Vector3 * out, const size_t N )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        out[idx] += c * a.cross( b[idx] );
    }
}
void add_c_cross( const scalar & c, const Vector3 & a, const vectorfield & b, vectorfield & out )
{
    unsigned int n = out.size();
    cu_add_c_cross<<<( n + 1023 ) / 1024, 1024>>>(
        c, a, raw_pointer_cast( b.data() ), raw_pointer_cast( out.data() ), n );
    CU_CHECK_AND_SYNC();
}

// out[i] += c * a[i] x b[i]
__global__ void cu_add_c_cross( const scalar c, const Vector3 * a, const Vector3 * b, Vector3 * out, const size_t N )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        out[idx] += c * a[idx].cross( b[idx] );
    }
}
void add_c_cross( const scalar & c, const vectorfield & a, const vectorfield & b, vectorfield & out )
{
    unsigned int n = out.size();
    cu_add_c_cross<<<( n + 1023 ) / 1024, 1024>>>(
        c, raw_pointer_cast( a.data() ), raw_pointer_cast( b.data() ), raw_pointer_cast( out.data() ), n );
    CU_CHECK_AND_SYNC();
}

// out[i] += c * a[i] x b[i]
__global__ void cu_add_c_cross( const scalar * c, const Vector3 * a, const Vector3 * b, Vector3 * out, const size_t N )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        out[idx] += c[idx] * a[idx].cross( b[idx] );
    }
}
void add_c_cross( const scalarfield & c, const vectorfield & a, const vectorfield & b, vectorfield & out )
{
    unsigned int n = out.size();
    cu_add_c_cross<<<( n + 1023 ) / 1024, 1024>>>(
        raw_pointer_cast( c.data() ), raw_pointer_cast( a.data() ), raw_pointer_cast( b.data() ),
        raw_pointer_cast( out.data() ), n );
    cudaDeviceSynchronize();
}

// out[i] = c * a x b[i]
__global__ void cu_set_c_cross( const scalar c, const Vector3 a, const Vector3 * b, Vector3 * out, const size_t N )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        out[idx] = c * a.cross( b[idx] );
    }
}
void set_c_cross( const scalar & c, const Vector3 & a, const vectorfield & b, vectorfield & out )
{
    unsigned int n = out.size();
    cu_set_c_cross<<<( n + 1023 ) / 1024, 1024>>>(
        c, a, raw_pointer_cast( b.data() ), raw_pointer_cast( out.data() ), n );
    CU_CHECK_AND_SYNC();
}

// out[i] = c * a[i] x b[i]
__global__ void cu_set_c_cross( const scalar c, const Vector3 * a, const Vector3 * b, Vector3 * out, const size_t N )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        out[idx] = c * a[idx].cross( b[idx] );
    }
}
void set_c_cross( const scalar & c, const vectorfield & a, const vectorfield & b, vectorfield & out )
{
    unsigned int n = out.size();
    cu_set_c_cross<<<( n + 1023 ) / 1024, 1024>>>(
        c, raw_pointer_cast( a.data() ), raw_pointer_cast( b.data() ), raw_pointer_cast( out.data() ), n );
    CU_CHECK_AND_SYNC();
}

} // namespace Vectormath
} // namespace Engine

#endif
