#pragma once
#ifndef SPIRIT_CORE_ENGINE_SOLVER_LBFGS_ATLAS_HPP
#define SPIRIT_CORE_ENGINE_SOLVER_LBFGS_ATLAS_HPP

#include <utility/Constants.hpp>
// #include <utility/Exception.hpp>
#include <engine/Backend_par.hpp>

#include <algorithm>

using namespace Utility;

template<>
inline void Method_Solver<Solver::LBFGS_Atlas>::Initialize()
{
    this->n_lbfgs_memory = 3; // how many updates the solver tracks to estimate the hessian
    this->atlas_updates  = std::vector<field<vector2field>>(
        this->noi, field<vector2field>( this->n_lbfgs_memory, vector2field( this->nos, { 0, 0 } ) ) );
    this->grad_atlas_updates = std::vector<field<vector2field>>(
        this->noi, field<vector2field>( this->n_lbfgs_memory, vector2field( this->nos, { 0, 0 } ) ) );
    this->rho                  = scalarfield( this->n_lbfgs_memory, 0 );
    this->alpha                = scalarfield( this->n_lbfgs_memory, 0 );
    this->forces               = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->forces_virtual       = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->atlas_coords3        = std::vector<scalarfield>( this->noi, scalarfield( this->nos, 1 ) );
    this->atlas_directions     = std::vector<vector2field>( this->noi, vector2field( this->nos, { 0, 0 } ) );
    this->atlas_residuals      = std::vector<vector2field>( this->noi, vector2field( this->nos, { 0, 0 } ) );
    this->atlas_residuals_last = std::vector<vector2field>( this->noi, vector2field( this->nos, { 0, 0 } ) );
    this->atlas_q_vec          = std::vector<vector2field>( this->noi, vector2field( this->nos, { 0, 0 } ) );
    this->maxmove              = 0.05;
    this->local_iter           = 0;

    for( int img = 0; img < this->noi; img++ )
    {
        // Choose atlas3 coordinates
        for( int i = 0; i < this->nos; i++ )
        {
            this->atlas_coords3[img][i] = ( *this->configurations[img] )[i][2] > 0 ? 1.0 : -1.0;
            // Solver_Kernels::ncg_spins_to_atlas( *this->configurations[i], this->atlas_coords[i], this->atlas_coords3[i] );
        }
    }
}

/*
    Stereographic coordinate system implemented according to an idea of F. Rybakov
    TODO: reference painless conjugate gradients
    See also Jorge Nocedal and Stephen J. Wright 'Numerical Optimization' Second Edition, 2006 (p. 121)
*/
template<>
inline void Method_Solver<Solver::LBFGS_Atlas>::Iteration()
{
    int noi = configurations.size();
    int nos = ( *configurations[0] ).size();

    // Current force
    this->Calculate_Force( this->configurations, this->forces );

    for( int img = 0; img < this->noi; img++ )
    {
        auto & image    = *this->configurations[img];
        auto & grad_ref = this->atlas_residuals[img];

        auto fv = this->forces_virtual[img].data();
        auto f  = this->forces[img].data();
        auto s  = image.data();

        Backend::par::apply( this->nos, [f, fv, s] SPIRIT_LAMBDA( int idx ) { fv[idx] = s[idx].cross( f[idx] ); } );

        Solver_Kernels::atlas_calc_gradients( grad_ref, image, this->forces[img], this->atlas_coords3[img] );
    }

    // Calculate search direction
    Solver_Kernels::lbfgs_get_searchdir(
        this->local_iter, this->rho, this->alpha, this->atlas_q_vec, this->atlas_directions, this->atlas_updates,
        this->grad_atlas_updates, this->atlas_residuals, this->atlas_residuals_last, this->n_lbfgs_memory, maxmove );

    scalar a_norm_rms = 0;
    // Scale by averaging
    for( int img = 0; img < noi; img++ )
    {
        a_norm_rms = std::max(
            a_norm_rms,
            scalar( sqrt(
                Backend::par::reduce(
                    this->atlas_directions[img], [] SPIRIT_LAMBDA( const Vector2 & v ) { return v.squaredNorm(); } )
                / nos ) ) );
    }
    scalar scaling = ( a_norm_rms > maxmove ) ? maxmove / a_norm_rms : 1.0;

    for( int img = 0; img < noi; img++ )
    {
        auto d = atlas_directions[img].data();
        Backend::par::apply( nos, [scaling, d] SPIRIT_LAMBDA( int idx ) { d[idx] *= scaling; } );
    }

    // Rotate spins
    Solver_Kernels::atlas_rotate( this->configurations, this->atlas_coords3, this->atlas_directions );

    if( Solver_Kernels::ncg_atlas_check_coordinates( this->configurations, this->atlas_coords3, -0.6 ) )
    {
        Solver_Kernels::lbfgs_atlas_transform_direction(
            this->configurations, this->atlas_coords3, this->atlas_updates, this->grad_atlas_updates,
            this->atlas_directions, this->atlas_residuals_last, this->rho );
    }
}

template<>
inline std::string Method_Solver<Solver::LBFGS_Atlas>::SolverName()
{
    return "LBFGS_Atlas";
}

template<>
inline std::string Method_Solver<Solver::LBFGS_Atlas>::SolverFullName()
{
    return "Limited memory Broyden-Fletcher-Goldfarb-Shanno using stereographic atlas";
}

#endif