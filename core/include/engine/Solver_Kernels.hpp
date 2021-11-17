#pragma once
#ifndef SPIRIT_CORE_ENGINE_SOLVER_KERNELS_HPP
#define SPIRIT_CORE_ENGINE_SOLVER_KERNELS_HPP

#include <memory>
#include <vector>

#include <Eigen/Core>
#include <complex>
#include <data/Spin_System.hpp>
#include <engine/Backend_par.hpp>
#include <engine/Backend_seq.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Vectormath_Defines.hpp>

namespace Engine
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
    std::vector<field<vector2field>> & atlas_updates, std::vector<field<vector2field>> & grad_updates,
    std::vector<vector2field> & searchdir, std::vector<vector2field> & grad_pr, scalarfield & rho );

// LBFGS
template<typename Vec>
void lbfgs_get_searchdir(
    int & local_iter, scalarfield & rho, scalarfield & alpha, std::vector<field<Vec>> & q_vec,
    std::vector<field<Vec>> & searchdir, std::vector<field<field<Vec>>> & delta_a,
    std::vector<field<field<Vec>>> & delta_grad, const std::vector<field<Vec>> & grad,
    std::vector<field<Vec>> & grad_pr, const int num_mem, const scalar maxmove )
{
    // std::cerr << "lbfgs searchdir \n";
    static auto dot = [] SPIRIT_LAMBDA( const Vec & v1, const Vec & v2 ) { return v1.dot( v2 ); };
    static auto set = [] SPIRIT_LAMBDA( const Vec & x ) { return x; };

    scalar epsilon = sizeof( scalar ) == sizeof( float ) ? 1e-30 : 1e-300;

    int noi     = grad.size();
    int nos     = grad[0].size();
    int m_index = local_iter % num_mem; // memory index
    int c_ind   = 0;

    if( local_iter == 0 ) // gradient descent
    {
        for( int img = 0; img < noi; img++ )
        {
            Backend::par::set( grad_pr[img], grad[img], set );
            auto & dir   = searchdir[img];
            auto & g_cur = grad[img];
            Backend::par::set( dir, g_cur, [] SPIRIT_LAMBDA( const Vec & x ) { return -x; } );
            auto & da = delta_a[img];
            auto & dg = delta_grad[img];
            for( int i = 0; i < num_mem; i++ )
            {
                rho[i]   = 0.0;
                auto dai = da[i].data();
                auto dgi = dg[i].data();
                Backend::par::apply(
                    nos,
                    [dai, dgi] SPIRIT_LAMBDA( int idx )
                    {
                        dai[idx] = Vec::Zero();
                        dgi[idx] = Vec::Zero();
                    } );
            }
        }
    }
    else
    {
        for( int img = 0; img < noi; img++ )
        {
            auto da   = delta_a[img][m_index].data();
            auto dg   = delta_grad[img][m_index].data();
            auto g    = grad[img].data();
            auto g_pr = grad_pr[img].data();
            auto sd   = searchdir[img].data();
            Backend::par::apply(
                nos,
                [da, dg, g, g_pr, sd] SPIRIT_LAMBDA( int idx )
                {
                    da[idx] = sd[idx];
                    dg[idx] = g[idx] - g_pr[idx];
                } );
        }

        scalar rinv_temp = 0;
        for( int img = 0; img < noi; img++ )
            rinv_temp += Backend::par::reduce( delta_grad[img][m_index], delta_a[img][m_index], dot );

        if( rinv_temp > epsilon )
            rho[m_index] = 1.0 / rinv_temp;
        else
        {
            local_iter = 0;
            return lbfgs_get_searchdir(
                local_iter, rho, alpha, q_vec, searchdir, delta_a, delta_grad, grad, grad_pr, num_mem, maxmove );
        }

        for( int img = 0; img < noi; img++ )
            Backend::par::set( q_vec[img], grad[img], set );

        for( int k = num_mem - 1; k > -1; k-- )
        {
            c_ind       = ( k + m_index + 1 ) % num_mem;
            scalar temp = 0;
            for( int img = 0; img < noi; img++ )
                temp += Backend::par::reduce( delta_a[img][c_ind], q_vec[img], dot );

            alpha[c_ind] = rho[c_ind] * temp;
            for( int img = 0; img < noi; img++ )
            {
                auto q = q_vec[img].data();
                auto a = alpha.data();
                auto d = delta_grad[img].data();
                Backend::par::apply(
                    nos, [c_ind, q, a, d] SPIRIT_LAMBDA( int idx ) { q[idx] += -a[c_ind] * d[c_ind][idx]; } );
            }
        }

        scalar dy2 = 0;
        for( int img = 0; img < noi; img++ )
            dy2 += Backend::par::reduce( delta_grad[img][m_index], delta_grad[img][m_index], dot );

        for( int img = 0; img < noi; img++ )
        {
            scalar rhody2     = dy2 * rho[m_index];
            scalar inv_rhody2 = 0.0;
            if( rhody2 > epsilon )
                inv_rhody2 = 1.0 / rhody2;
            else
                inv_rhody2 = 1.0 / ( epsilon );
            Backend::par::set(
                searchdir[img], q_vec[img], [inv_rhody2] SPIRIT_LAMBDA( const Vec & q ) { return inv_rhody2 * q; } );
        }

        for( int k = 0; k < num_mem; k++ )
        {
            if( local_iter < num_mem )
                c_ind = k;
            else
                c_ind = ( k + m_index + 1 ) % num_mem;

            scalar rhopdg = 0;
            for( int img = 0; img < noi; img++ )
                rhopdg += Backend::par::reduce( delta_grad[img][c_ind], searchdir[img], dot );

            rhopdg *= rho[c_ind];

            for( int img = 0; img < noi; img++ )
            {
                auto sd   = searchdir[img].data();
                auto alph = alpha[c_ind];
                auto da   = delta_a[img][c_ind].data();
                Backend::par::apply(
                    nos, [sd, alph, da, rhopdg] SPIRIT_LAMBDA( int idx ) { sd[idx] += ( alph - rhopdg ) * da[idx]; } );
            }
        }

        for( int img = 0; img < noi; img++ )
        {
            auto g    = grad[img].data();
            auto g_pr = grad_pr[img].data();
            auto sd   = searchdir[img].data();
            Backend::par::apply(
                nos,
                [g, g_pr, sd] SPIRIT_LAMBDA( int idx )
                {
                    g_pr[idx] = g[idx];
                    sd[idx]   = -sd[idx];
                } );
        }
    }
    local_iter++;
}

} // namespace Solver_Kernels
} // namespace Engine

#endif