#include <Eigen/Dense>

#include <engine/Backend.hpp>
#include <engine/Solver_Kernels.hpp>
#include <utility/Constants.hpp>

using namespace Utility;
using Utility::Constants::Pi;

namespace Engine
{

namespace Solver_Kernels
{

void sib_transform( const vectorfield & spins, const vectorfield & force, vectorfield & out )
{
    const int n = spins.size();

    const auto * s = spins.data();
    const auto * f = force.data();
    auto * o       = out.data();

    Backend::for_each_n(
        SPIRIT_PAR Backend::make_counting_iterator( 0 ), n,
        [s, f, o] SPIRIT_LAMBDA( const int idx )
        {
            const Vector3 e1   = s[idx];
            const Vector3 A    = 0.5 * f[idx];
            const scalar Anorm = A.norm();
            // 1/determinant(A)
            const scalar detAi = 1.0 / ( 1 + Anorm * Anorm );

            // calculate equation witho the predictor?
            const Vector3 a2 = e1 - e1.cross( A );

            o[idx][0]
                = ( a2[0] * ( A[0] * A[0] + 1 ) + a2[1] * ( A[0] * A[1] - A[2] ) + a2[2] * ( A[0] * A[2] + A[1] ) )
                  * detAi;
            o[idx][1]
                = ( a2[0] * ( A[1] * A[0] + A[2] ) + a2[1] * ( A[1] * A[1] + 1 ) + a2[2] * ( A[1] * A[2] - A[0] ) )
                  * detAi;
            o[idx][2]
                = ( a2[0] * ( A[2] * A[0] - A[1] ) + a2[1] * ( A[2] * A[1] + A[0] ) + a2[2] * ( A[2] * A[2] + 1 ) )
                  * detAi;
        } );
}

void oso_calc_gradients( vectorfield & grad, const vectorfield & spins, const vectorfield & forces )
{
    const Matrix3 t = ( Matrix3() << 0, 0, 1, 0, -1, 0, 1, 0, 0 ).finished();

    const auto * s = spins.data();
    const auto * f = forces.data();
    auto * g       = grad.data();

    Backend::for_each_n(
        SPIRIT_PAR Backend::make_counting_iterator( 0 ), spins.size(),
        [s, f, t, g] SPIRIT_LAMBDA( const int idx ) { g[idx] = t * ( -s[idx].cross( f[idx] ) ); } );
}

void oso_rotate( vectorfield & spins, const vectorfield & searchdir )
{
    auto * s        = spins.data();
    const auto * sd = searchdir.data();

    Backend::for_each_n(
        SPIRIT_PAR Backend::make_counting_iterator( 0 ), spins.size(),
        [s, sd] SPIRIT_LAMBDA( const int idx )
        {
            const scalar theta = ( sd[idx] ).norm();
            const scalar q = cos( theta ), w = 1 - q, x = -sd[idx][0] / theta, y = -sd[idx][1] / theta,
                         z = -sd[idx][2] / theta, s1 = -y * z * w, s2 = x * z * w, s3 = -x * y * w,
                         p1 = x * sin( theta ), p2 = y * sin( theta ), p3 = z * sin( theta );

            if( theta > 1.0e-20 ) // if theta is too small we do nothing
            {
                const scalar t1 = ( q + z * z * w ) * s[idx][0] + ( s1 + p1 ) * s[idx][1] + ( s2 + p2 ) * s[idx][2];
                const scalar t2 = ( s1 - p1 ) * s[idx][0] + ( q + y * y * w ) * s[idx][1] + ( s3 + p3 ) * s[idx][2];
                const scalar t3 = ( s2 - p2 ) * s[idx][0] + ( s3 - p3 ) * s[idx][1] + ( q + x * x * w ) * s[idx][2];
                s[idx][0]       = t1;
                s[idx][1]       = t2;
                s[idx][2]       = t3;
            };
        } );
}

scalar maximum_rotation( const vectorfield & searchdir, scalar maxmove )
{
    int nos          = searchdir.size();
    scalar theta_rms = 0;
    theta_rms        = sqrt(
        Backend::transform_reduce(
            SPIRIT_PAR searchdir.begin(), searchdir.end(), scalar( 0.0 ), Backend::plus<scalar>{},
            [] SPIRIT_LAMBDA( const Vector3 & v ) { return v.squaredNorm(); } )
        / nos );
    scalar scaling = ( theta_rms > maxmove ) ? maxmove / theta_rms : 1.0;
    return scaling;
}

void atlas_rotate( vectorfield & spins, const scalarfield & a3_coords, const vector2field & searchdir )
{
    auto * s        = spins.data();
    const auto * d  = searchdir.data();
    const auto * a3 = a3_coords.data();
    Backend::for_each_n(
        SPIRIT_PAR Backend::make_counting_iterator( 0 ), spins.size(),
        [s, d, a3] SPIRIT_LAMBDA( const int idx )
        {
            const scalar gamma = ( 1 + s[idx][2] * a3[idx] );
            const scalar denom = ( s[idx].head<2>().squaredNorm() ) / gamma + 2 * d[idx].dot( s[idx].head<2>() )
                                 + gamma * d[idx].squaredNorm();
            s[idx].head<2>() = 2 * ( s[idx].head<2>() + d[idx] * gamma );
            s[idx][2]        = a3[idx] * ( gamma - denom );
            s[idx] *= 1 / ( gamma + denom );
        } );
}

void atlas_calc_gradients(
    vector2field & residuals, const vectorfield & spins, const vectorfield & forces, const scalarfield & a3_coords )
{
    const auto * s  = spins.data();
    const auto * a3 = a3_coords.data();
    const auto * f  = forces.data();
    auto * g        = residuals.data();

    Backend::for_each_n(
        SPIRIT_PAR Backend::make_counting_iterator( 0 ), spins.size(),
        [s, a3, f, g] SPIRIT_LAMBDA( const int idx )
        {
            const scalar J00 = s[idx][1] * s[idx][1] + s[idx][2] * ( s[idx][2] + a3[idx] );
            const scalar J10 = -s[idx][0] * s[idx][1];
            const scalar J01 = -s[idx][0] * s[idx][1];
            const scalar J11 = s[idx][0] * s[idx][0] + s[idx][2] * ( s[idx][2] + a3[idx] );
            const scalar J02 = -s[idx][0] * ( s[idx][2] + a3[idx] );
            const scalar J12 = -s[idx][1] * ( s[idx][2] + a3[idx] );

            g[idx][0] = -( J00 * f[idx][0] + J01 * f[idx][1] + J02 * f[idx][2] );
            g[idx][1] = -( J10 * f[idx][0] + J11 * f[idx][1] + J12 * f[idx][2] );
        } );
}

bool ncg_atlas_check_coordinates( const vectorfield & spins, const scalarfield & a3_coords, const scalar tol )
{
    // CUDA doesn't like reducing over bools, so we are using ints instead.
    auto result = Backend::transform_reduce(
        SPIRIT_PAR spins.begin(), spins.end(), a3_coords.begin(),
#ifndef SPIRIT_USE_CUDA
        false, [] SPIRIT_LAMBDA( const bool lhs, const bool rhs ) { return lhs || rhs; },
        [tol] SPIRIT_LAMBDA( const Vector3 & s, const scalar a3 ) { return s[2] * a3 < tol; }
#else
        0, [] SPIRIT_LAMBDA( const int lhs, const int rhs ) { return ( lhs == 0 && rhs == 0 ) ? 0 : 1; },
        [tol] SPIRIT_LAMBDA( const Vector3 & s, const scalar a3 ) -> int { return s[2] * a3 < tol ? 1 : 0; }
#endif
    );

#ifndef SPIRIT_USE_CUDA
    return result;
#else
    return result != 0;
#endif
}

void lbfgs_atlas_transform_direction(
    const vectorfield & spins, scalarfield & a3_coords, std::vector<vector2field> & atlas_updates,
    std::vector<vector2field> & grad_updates, vector2field & searchdir, vector2field & grad_pr, scalarfield & rho_inv )
{
    const auto * s = spins.data();
    auto * a3      = a3_coords.data();
    auto * sd      = searchdir.data();
    auto * g_pr    = grad_pr.data();
    auto * rh      = rho_inv.data();

    const auto n_mem = atlas_updates.size();

    field<Vector2 *> t1( n_mem ), t2( n_mem );
    for( int n = 0; n < n_mem; n++ )
    {
        t1[n] = atlas_updates[n].data();
        t2[n] = grad_updates[n].data();
    }

    auto * const * const a_up = t1.data();
    auto * const * const g_up = t2.data();

    Backend::for_each_n(
        SPIRIT_PAR Backend::make_counting_iterator( 0 ), spins.size(),
        [s, a3, sd, g_pr, rh, a_up, g_up, n_mem] SPIRIT_LAMBDA( const int idx )
        {
            if( s[idx][2] * a3[idx] < 0 )
            {
                // Transform coordinates to optimal map
                a3[idx]             = ( s[idx][2] > 0 ) ? 1 : -1;
                const scalar factor = ( 1 - a3[idx] * s[idx][2] ) / ( 1 + a3[idx] * s[idx][2] );
                sd[idx] *= factor;
                g_pr[idx] *= factor;

                for( int n = 0; n < n_mem; n++ )
                {
                    rh[n] += ( factor * factor - 1 ) * a_up[n][idx].dot( g_up[n][idx] );
                    a_up[n][idx] *= factor;
                    g_up[n][idx] *= factor;
                }
            }
        } );
}

} // namespace Solver_Kernels

} // namespace Engine
