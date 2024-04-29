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

} // namespace Vectormath
} // namespace Engine

#endif
