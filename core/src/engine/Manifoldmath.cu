#ifdef SPIRIT_USE_CUDA

#include <engine/Backend.hpp>
#include <engine/Manifoldmath.hpp>
#include <engine/Vectormath.hpp>
#include <utility/Constants.hpp>

#include <Eigen/Dense>

#include <cmath>

namespace C = Utility::Constants;

// CUDA Version
namespace Engine
{
namespace Manifoldmath
{

void project_parallel( vectorfield & vf1, const vectorfield & vf2 )
{
    scalar proj = Vectormath::dot( vf1, vf2 );
    Backend::for_each_n(
        SPIRIT_PAR Backend::make_counting_iterator( 0 ), vf1.size(),
        [vf1 = raw_pointer_cast( vf1.data() ), vf2 = raw_pointer_cast( vf2.data() ), proj] SPIRIT_LAMBDA( int idx )
        { vf1[idx] = proj * vf2[idx]; } );
}

__global__ void cu_project_orthogonal( Vector3 * vf1, const Vector3 * vf2, scalar proj, size_t N )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if( idx < N )
    {
        vf1[idx] -= proj * vf2[idx];
    }
}
// The wrapper for the calling of the actual kernel
void project_orthogonal( vectorfield & vf1, const vectorfield & vf2 )
{
    int n = vf1.size();

    // Get projection
    scalar proj = Vectormath::dot( vf1, vf2 );
    // Project vf1
    cu_project_orthogonal<<<( n + 1023 ) / 1024, 1024>>>(
        raw_pointer_cast( vf1.data() ), raw_pointer_cast( vf2.data() ), proj, n );
    CU_CHECK_AND_SYNC();
}

void invert_parallel( vectorfield & vf1, const vectorfield & vf2 )
{
    scalar proj = Vectormath::dot( vf1, vf2 );
    Vectormath::add_c_a( -2 * proj, vf2, vf1 );
}

void invert_orthogonal( vectorfield & vf1, const vectorfield & vf2 )
{
    vectorfield vf3 = vf1;
    project_orthogonal( vf3, vf2 );
    Vectormath::add_c_a( -2, vf3, vf1 );
}

__global__ void cu_project_tangential( Vector3 * vf1, const Vector3 * vf2, size_t N )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if( idx < N )
    {
        vf1[idx] -= vf1[idx].dot( vf2[idx] ) * vf2[idx];
    }
}
void project_tangential( vectorfield & vf1, const vectorfield & vf2 )
{
    int n = vf1.size();
    cu_project_tangential<<<( n + 1023 ) / 1024, 1024>>>(
        raw_pointer_cast( vf1.data() ), raw_pointer_cast( vf2.data() ), n );
    CU_CHECK_AND_SYNC();
}

__inline__ __device__ scalar cu_dist_greatcircle( const Vector3 v1, const Vector3 v2 )
{
    scalar r = v1.dot( v2 );

    // Prevent NaNs from occurring
    r = max( -1.0, min( 1.0, r ) );

    // Greatcircle distance
    return std::acos( r );
}
scalar dist_greatcircle( const Vector3 & v1, const Vector3 & v2 )
{
    scalar r = v1.dot( v2 );

    // Prevent NaNs from occurring
    r = std::max( scalar( -1 ), std::min( scalar( 1 ), r ) );

    // Greatcircle distance
    return std::acos( r );
}

// Calculates the squares of the geodesic distances between vectors of two vectorfields
__global__ void cu_dist_geodesic_2( const Vector3 * vf1, const Vector3 * vf2, scalar * sf, int N )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        scalar d = cu_dist_greatcircle( vf1[idx], vf2[idx] );
        sf[idx]  = d * d;
    }
}
scalar dist_geodesic( const vectorfield & vf1, const vectorfield & vf2 )
{
    int n = vf1.size();
    scalarfield sf( n );

    cu_dist_geodesic_2<<<( n + 1023 ) / 1024, 1024>>>(
        raw_pointer_cast( vf1.data() ), raw_pointer_cast( vf2.data() ), raw_pointer_cast( sf.data() ), n );
    CU_CHECK_AND_SYNC();

    scalar dist = Vectormath::sum( sf );
    return sqrt( dist );
}

/*
Calculates the 'tangent' vectors, i.e.in crudest approximation the difference between an image and the neighbouring
*/
void Tangents(
    std::vector<std::shared_ptr<vectorfield>> configurations, const std::vector<scalar> & energies,
    std::vector<vectorfield> & tangents )
{
    int noi = configurations.size();
    int nos = ( *configurations[0] ).size();

    for( int idx_img = 0; idx_img < noi; ++idx_img )
    {
        auto & image = *configurations[idx_img];

        // First Image
        if( idx_img == 0 )
        {
            auto & image_plus = *configurations[idx_img + 1];
            Vectormath::set_c_a( 1, image_plus, tangents[idx_img] );
            Vectormath::add_c_a( -1, image, tangents[idx_img] );
        }
        // Last Image
        else if( idx_img == noi - 1 )
        {
            auto & image_minus = *configurations[idx_img - 1];
            Vectormath::set_c_a( 1, image, tangents[idx_img] );
            Vectormath::add_c_a( -1, image_minus, tangents[idx_img] );
        }
        // Images Inbetween
        else
        {
            auto & image_plus  = *configurations[idx_img + 1];
            auto & image_minus = *configurations[idx_img - 1];

            // Energies
            scalar E_mid = 0, E_plus = 0, E_minus = 0;
            E_mid   = energies[idx_img];
            E_plus  = energies[idx_img + 1];
            E_minus = energies[idx_img - 1];

            // Vectors to neighbouring images
            vectorfield t_plus( nos ), t_minus( nos );

            Vectormath::set_c_a( 1, image_plus, t_plus );
            Vectormath::add_c_a( -1, image, t_plus );

            Vectormath::set_c_a( 1, image, t_minus );
            Vectormath::add_c_a( -1, image_minus, t_minus );

            // Near maximum or minimum
            if( ( E_plus < E_mid && E_mid > E_minus ) || ( E_plus > E_mid && E_mid < E_minus ) )
            {
                // Get a smooth transition between forward and backward tangent
                scalar E_max = std::max( std::abs( E_plus - E_mid ), std::abs( E_minus - E_mid ) );
                scalar E_min = std::min( std::abs( E_plus - E_mid ), std::abs( E_minus - E_mid ) );

                if( E_plus > E_minus )
                {
                    Vectormath::set_c_a( E_max, t_plus, tangents[idx_img] );
                    Vectormath::add_c_a( E_min, t_minus, tangents[idx_img] );
                }
                else
                {
                    Vectormath::set_c_a( E_min, t_plus, tangents[idx_img] );
                    Vectormath::add_c_a( E_max, t_minus, tangents[idx_img] );
                }
            }
            // Rising slope
            else if( E_plus > E_mid && E_mid > E_minus )
            {
                Vectormath::set_c_a( 1, t_plus, tangents[idx_img] );
            }
            // Falling slope
            else if( E_plus < E_mid && E_mid < E_minus )
            {
                Vectormath::set_c_a( 1, t_minus, tangents[idx_img] );
                // tangents = t_minus;
                for( int i = 0; i < nos; ++i )
                {
                    tangents[idx_img][i] = t_minus[i];
                }
            }
            // No slope(constant energy)
            else
            {
                Vectormath::set_c_a( 1, t_plus, tangents[idx_img] );
                Vectormath::add_c_a( 1, t_minus, tangents[idx_img] );
            }
        }

        // Project tangents into tangent planes of spin vectors to make them actual tangents
        project_tangential( tangents[idx_img], image );

        // Normalise in 3N - dimensional space
        Manifoldmath::normalize( tangents[idx_img] );

    } // end for idx_img
} // end Tangents

} // namespace Manifoldmath
} // namespace Engine

#endif
