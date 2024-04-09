#pragma once
#ifndef SPIRIT_CORE_ENGINE_SPIN_SOLVER_KERNELS_HPP
#define SPIRIT_CORE_ENGINE_SPIN_SOLVER_KERNELS_HPP

#include <memory>
#include <vector>

#include <Eigen/Core>
#include <complex>
#include <data/Spin_System.hpp>
#include <engine/Backend.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Vectormath_Defines.hpp>

namespace Engine
{

namespace Spin
{

namespace Solver_Kernels
{

// SIB
void sib_transform( const vectorfield & spins, const vectorfield & force, vectorfield & out );

// OSO coordinates
void oso_rotate( std::vector<std::shared_ptr<vectorfield>> & configurations, std::vector<vectorfield> & searchdir );
void oso_calc_gradients( vectorfield & residuals, const vectorfield & spins, const vectorfield & forces );
scalar maximum_rotation( const vectorfield & searchdir, scalar maxmove );

// Atlas coordinates
void atlas_calc_gradients(
    vector2field & residuals, const vectorfield & spins, const vectorfield & forces, const scalarfield & a3_coords );
void atlas_rotate(
    std::vector<std::shared_ptr<vectorfield>> & configurations, const std::vector<scalarfield> & a3_coords,
    const std::vector<vector2field> & searchdir );
bool ncg_atlas_check_coordinates(
    const std::vector<std::shared_ptr<vectorfield>> & spins, std::vector<scalarfield> & a3_coords, scalar tol = -0.6 );
void lbfgs_atlas_transform_direction(
    std::vector<std::shared_ptr<vectorfield>> & configurations, std::vector<scalarfield> & a3_coords,
    std::vector<std::vector<vector2field>> & atlas_updates, std::vector<std::vector<vector2field>> & grad_updates,
    std::vector<vector2field> & searchdir, std::vector<vector2field> & grad_pr, scalarfield & rho );

// LBFGS
template<typename Vec>
void lbfgs_get_searchdir(
    int & local_iter, scalarfield & rho, scalarfield & alpha, std::vector<field<Vec>> & q_vec,
    std::vector<field<Vec>> & searchdir, std::vector<std::vector<field<Vec>>> & delta_a,
    std::vector<std::vector<field<Vec>>> & delta_grad, const std::vector<field<Vec>> & grad,
    std::vector<field<Vec>> & grad_pr, const int num_mem, const scalar maxmove )
{
    using std::begin, std::end;

    // std::cerr << "lbfgs searchdir \n";
    static auto dot     = [] SPIRIT_HOSTDEVICE( const Vec & v1, const Vec & v2 ) { return v1.dot( v2 ); };
    static auto set     = [] SPIRIT_HOSTDEVICE( const Vec & x ) { return x; };
    static auto inverse = [] SPIRIT_HOSTDEVICE( const Vec & x ) { return -x; };

    scalar epsilon = sizeof( scalar ) == sizeof( float ) ? 1e-30 : 1e-300;

    int noi     = grad.size();
    int m_index = local_iter % num_mem; // memory index
    int c_ind   = 0;

    if( local_iter == 0 ) // gradient descent
    {
        for( int img = 0; img < noi; img++ )
        {
            Backend::copy( SPIRIT_PAR begin( grad[img] ), end( grad[img] ), begin( grad_pr[img] ) );
            Backend::transform( SPIRIT_PAR begin( grad[img] ), end( grad[img] ), begin( searchdir[img] ), inverse );

            auto & da = delta_a[img];
            auto & dg = delta_grad[img];
            Backend::fill_n( begin( rho ), num_mem, 0.0 );
            for( int i = 0; i < num_mem; i++ )
            {
                Backend::fill( begin( delta_a[img][i] ), end( delta_a[img][i] ), Vec::Zero() );
                Backend::fill( begin( delta_grad[img][i] ), end( delta_grad[img][i] ), Vec::Zero() );
            }
        }
    }
    else
    {
        for( int img = 0; img < noi; img++ )
        {
            Backend::copy( SPIRIT_PAR begin( searchdir[img] ), end( searchdir[img] ), begin( delta_a[img][m_index] ) );
            Backend::transform(
                SPIRIT_PAR begin( grad[img] ), end( grad[img] ), begin( grad_pr[img] ),
                begin( delta_grad[img][m_index] ),
                [] SPIRIT_HOSTDEVICE( const Vec & g, const Vec & g_pr ) { return g - g_pr; } );
        }

        scalar rinv_temp = 0;
        for( int img = 0; img < noi; img++ )
            rinv_temp += Backend::transform_reduce(
                SPIRIT_PAR begin( delta_grad[img][m_index] ), end( delta_grad[img][m_index] ),
                begin( delta_a[img][m_index] ), scalar( 0 ), Backend::plus<scalar>{}, dot );

        if( rinv_temp > epsilon )
            rho[m_index] = 1.0 / rinv_temp;
        else
        {
            local_iter = 0;
            return lbfgs_get_searchdir(
                local_iter, rho, alpha, q_vec, searchdir, delta_a, delta_grad, grad, grad_pr, num_mem, maxmove );
        }

        for( int img = 0; img < noi; img++ )
            Backend::copy( SPIRIT_PAR begin( grad[img] ), end( grad[img] ), begin( q_vec[img] ) );

        for( int k = num_mem - 1; k > -1; k-- )
        {
            c_ind       = ( k + m_index + 1 ) % num_mem;
            scalar temp = 0;
            for( int img = 0; img < noi; img++ )
                temp += Backend::transform_reduce(
                    SPIRIT_PAR begin( delta_a[img][c_ind] ), end( delta_a[img][c_ind] ), begin( q_vec[img] ),
                    scalar( 0 ), Backend::plus<scalar>{}, dot );

            alpha[c_ind] = rho[c_ind] * temp;
            for( int img = 0; img < noi; img++ )
            {
                auto a = alpha[c_ind];
                Backend::transform(
                    SPIRIT_PAR begin( q_vec[img] ), end( q_vec[img] ), begin( delta_grad[img][c_ind] ),
                    begin( q_vec[img] ),
                    [a] SPIRIT_HOSTDEVICE( const Vec & q, const Vec & d ) -> Vec { return q - a * d; } );
            }
        }

        scalar dy2 = 0;
        for( int img = 0; img < noi; img++ )
            dy2 += Backend::transform_reduce(
                SPIRIT_PAR begin( delta_grad[img][m_index] ), end( delta_grad[img][m_index] ),
                begin( delta_grad[img][m_index] ), scalar( 0 ), Backend::plus<scalar>{}, dot );

        for( int img = 0; img < noi; img++ )
        {
            scalar rhody2     = dy2 * rho[m_index];
            scalar inv_rhody2 = 0.0;
            if( rhody2 > epsilon )
                inv_rhody2 = 1.0 / rhody2;
            else
                inv_rhody2 = 1.0 / ( epsilon );
            Backend::transform(
                SPIRIT_PAR begin( q_vec[img] ), end( q_vec[img] ), begin( searchdir[img] ),
                [inv_rhody2] SPIRIT_HOSTDEVICE( const Vec & q ) { return inv_rhody2 * q; } );
        }

        for( int k = 0; k < num_mem; k++ )
        {
            if( local_iter < num_mem )
                c_ind = k;
            else
                c_ind = ( k + m_index + 1 ) % num_mem;

            scalar rhopdg = 0;
            for( int img = 0; img < noi; img++ )
                rhopdg += Backend::transform_reduce(
                    SPIRIT_PAR begin( delta_grad[img][c_ind] ), end( delta_grad[img][c_ind] ), begin( searchdir[img] ),
                    scalar( 0 ), Backend::plus<scalar>{}, dot );

            rhopdg *= rho[c_ind];

            for( int img = 0; img < noi; img++ )
            {
                const auto alph = alpha[c_ind] - rhopdg;
                Backend::transform(
                    SPIRIT_PAR begin( searchdir[img] ), end( searchdir[img] ), begin( delta_a[img][c_ind] ),
                    begin( searchdir[img] ),
                    [alph] SPIRIT_HOSTDEVICE( const Vec & sd, const Vec & da ) { return sd + alph * da; } );
            }
        }

        for( int img = 0; img < noi; img++ )
        {
            auto g    = grad[img].data();
            auto g_pr = grad_pr[img].data();
            auto sd   = searchdir[img].data();

            Backend::transform(
                SPIRIT_PAR begin( searchdir[img] ), end( searchdir[img] ), begin( searchdir[img] ), inverse );
            Backend::copy( SPIRIT_PAR begin( grad[img] ), end( grad[img] ), begin( grad_pr[img] ) );
        }
    }
    local_iter++;
}

} // namespace Solver_Kernels

} // namespace Spin

} // namespace Engine

#endif
