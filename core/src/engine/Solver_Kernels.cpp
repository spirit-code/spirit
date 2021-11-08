#include <Eigen/Dense>

#include <engine/Backend_par.hpp>
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
    int n = spins.size();

    auto s = spins.data();
    auto f = force.data();
    auto o = out.data();

    Backend::par::apply(
        n,
        [s, f, o] SPIRIT_LAMBDA( int idx )
        {
            Vector3 e1, a2, A;
            scalar detAi;
            e1 = s[idx];
            A  = 0.5 * f[idx];

            // 1/determinant(A)
            detAi = 1.0 / ( 1 + pow( A.norm(), 2.0 ) );

            // calculate equation witho the predictor?
            a2 = e1 - e1.cross( A );

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

    auto g = grad.data();
    auto s = spins.data();
    auto f = forces.data();

    Backend::par::apply(
        spins.size(), [g, s, f, t] SPIRIT_LAMBDA( int idx ) { g[idx] = t * ( -s[idx].cross( f[idx] ) ); } );
}

void oso_rotate( std::vector<std::shared_ptr<vectorfield>> & configurations, std::vector<vectorfield> & searchdir )
{
    int noi = configurations.size();
    int nos = configurations[0]->size();
    for( int img = 0; img < noi; ++img )
    {

        auto s  = configurations[img]->data();
        auto sd = searchdir[img].data();

        Backend::par::apply(
            nos,
            [s, sd] SPIRIT_LAMBDA( int idx )
            {
                scalar theta = ( sd[idx] ).norm();
                scalar q = cos( theta ), w = 1 - q, x = -sd[idx][0] / theta, y = -sd[idx][1] / theta,
                       z = -sd[idx][2] / theta, s1 = -y * z * w, s2 = x * z * w, s3 = -x * y * w, p1 = x * sin( theta ),
                       p2 = y * sin( theta ), p3 = z * sin( theta );

                scalar t1, t2, t3;
                if( theta > 1.0e-20 ) // if theta is too small we do nothing
                {
                    t1        = ( q + z * z * w ) * s[idx][0] + ( s1 + p1 ) * s[idx][1] + ( s2 + p2 ) * s[idx][2];
                    t2        = ( s1 - p1 ) * s[idx][0] + ( q + y * y * w ) * s[idx][1] + ( s3 + p3 ) * s[idx][2];
                    t3        = ( s2 - p2 ) * s[idx][0] + ( s3 - p3 ) * s[idx][1] + ( q + x * x * w ) * s[idx][2];
                    s[idx][0] = t1;
                    s[idx][1] = t2;
                    s[idx][2] = t3;
                };
            } );
    }
}

scalar maximum_rotation( const vectorfield & searchdir, scalar maxmove )
{
    int nos          = searchdir.size();
    scalar theta_rms = 0;
    theta_rms        = sqrt(
        Backend::par::reduce( searchdir, [] SPIRIT_LAMBDA( const Vector3 & v ) { return v.squaredNorm(); } ) / nos );
    scalar scaling = ( theta_rms > maxmove ) ? maxmove / theta_rms : 1.0;
    return scaling;
}

void atlas_rotate(
    std::vector<std::shared_ptr<vectorfield>> & configurations, const std::vector<scalarfield> & a3_coords,
    const std::vector<vector2field> & searchdir )
{
    int noi = configurations.size();
    int nos = configurations[0]->size();
    for( int img = 0; img < noi; img++ )
    {
        auto spins = configurations[img]->data();
        auto d     = searchdir[img].data();
        auto a3    = a3_coords[img].data();
        Backend::par::apply(
            nos,
            [nos, spins, d, a3] SPIRIT_LAMBDA( int idx )
            {
                const scalar gamma = ( 1 + spins[idx][2] * a3[idx] );
                const scalar denom = ( spins[idx].head<2>().squaredNorm() ) / gamma
                                     + 2 * d[idx].dot( spins[idx].head<2>() ) + gamma * d[idx].squaredNorm();
                spins[idx].head<2>() = 2 * ( spins[idx].head<2>() + d[idx] * gamma );
                spins[idx][2]        = a3[idx] * ( gamma - denom );
                spins[idx] *= 1 / ( gamma + denom );
            } );
    }
}

void atlas_calc_gradients(
    vector2field & residuals, const vectorfield & spins, const vectorfield & forces, const scalarfield & a3_coords )
{
    auto s  = spins.data();
    auto a3 = a3_coords.data();
    auto g  = residuals.data();
    auto f  = forces.data();

    Backend::par::apply(
        spins.size(),
        [s, a3, g, f] SPIRIT_LAMBDA( int idx )
        {
            scalar J00 = s[idx][1] * s[idx][1] + s[idx][2] * ( s[idx][2] + a3[idx] );
            scalar J10 = -s[idx][0] * s[idx][1];
            scalar J01 = -s[idx][0] * s[idx][1];
            scalar J11 = s[idx][0] * s[idx][0] + s[idx][2] * ( s[idx][2] + a3[idx] );
            scalar J02 = -s[idx][0] * ( s[idx][2] + a3[idx] );
            scalar J12 = -s[idx][1] * ( s[idx][2] + a3[idx] );

            g[idx][0] = -( J00 * f[idx][0] + J01 * f[idx][1] + J02 * f[idx][2] );
            g[idx][1] = -( J10 * f[idx][0] + J11 * f[idx][1] + J12 * f[idx][2] );
        } );
}

bool ncg_atlas_check_coordinates(
    const std::vector<std::shared_ptr<vectorfield>> & spins, std::vector<scalarfield> & a3_coords, scalar tol )
{
    int noi = spins.size();
    int nos = ( *spins[0] ).size();

    // We use `int` instead of `bool`, because somehow cuda does not like pointers to bool
    // TODO: fix in future
    field<int> result = field<int>( 1, int( false ) );

    for( int img = 0; img < noi; img++ )
    {
        auto s    = spins[0]->data();
        auto a3   = a3_coords[img].data();
        int * res = &result[0];

        Backend::par::apply(
            nos,
            [s, a3, tol, res] SPIRIT_LAMBDA( int idx )
            {
                if( s[idx][2] * a3[idx] < tol && res[0] == int( false ) )
                    res[0] = int( true );
            } );
    }

    return bool( result[0] );
}

void lbfgs_atlas_transform_direction(
    std::vector<std::shared_ptr<vectorfield>> & configurations, std::vector<scalarfield> & a3_coords,
    std::vector<field<vector2field>> & atlas_updates, std::vector<field<vector2field>> & grad_updates,
    std::vector<vector2field> & searchdir, std::vector<vector2field> & grad_pr, scalarfield & rho )
{
    int noi = configurations.size();
    int nos = configurations[0]->size();

    for( int n = 0; n < atlas_updates[0].size(); n++ )
    {
        rho[n] = 1 / rho[n];
    }

    for( int img = 0; img < noi; img++ )
    {
        auto s    = ( *configurations[img] ).data();
        auto a3   = a3_coords[img].data();
        auto sd   = searchdir[img].data();
        auto g_pr = grad_pr[img].data();
        auto rh   = rho.data();

        auto n_mem = atlas_updates[img].size();

        field<Vector2 *> t1( n_mem ), t2( n_mem );
        for( int n = 0; n < n_mem; n++ )
        {
            t1[n] = ( atlas_updates[img][n].data() );
            t2[n] = ( grad_updates[img][n].data() );
        }

        auto a_up = t1.data();
        auto g_up = t2.data();

        Backend::par::apply(
            nos,
            [s, a3, sd, g_pr, rh, a_up, g_up, n_mem] SPIRIT_LAMBDA( int idx )
            {
                scalar factor = 1;
                if( s[idx][2] * a3[idx] < 0 )
                {
                    // Transform coordinates to optimal map
                    a3[idx] = ( s[idx][2] > 0 ) ? 1 : -1;
                    factor  = ( 1 - a3[idx] * s[idx][2] ) / ( 1 + a3[idx] * s[idx][2] );
                    sd[idx] *= factor;
                    g_pr[idx] *= factor;

                    for( int n = 0; n < n_mem; n++ )
                    {
                        rh[n] = rh[n] + ( factor * factor - 1 ) * a_up[n][idx].dot( g_up[n][idx] );
                        a_up[n][idx] *= factor;
                        g_up[n][idx] *= factor;
                    }
                }
            } );
    }

    for( int n = 0; n < atlas_updates[0].size(); n++ )
    {
        rho[n] = 1 / rho[n];
    }
}

} // namespace Solver_Kernels
} // namespace Engine