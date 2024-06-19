#pragma once
#ifndef SPIRIT_CORE_ENGINE_SPIN_SOLVER_LBFGS_ATLAS_HPP
#define SPIRIT_CORE_ENGINE_SPIN_SOLVER_LBFGS_ATLAS_HPP

#include <engine/Backend.hpp>
#include <engine/spin/Method_Solver.hpp>
#include <utility/Constants.hpp>

#include <utility>

namespace Engine
{

namespace Spin
{

template<>
class SolverData<Solver::LBFGS_Atlas> : public SolverMethods
{
protected:
    using SolverMethods::Calculate_Force;
    using SolverMethods::Calculate_Force_Virtual;
    using SolverMethods::Prepare_Thermal_Field;
    using SolverMethods::SolverMethods;
    // General
    static constexpr int n_lbfgs_memory = 3; // how many updates the solver tracks to estimate the hessian
    static constexpr scalar maxmove     = 0.05;
    int local_iter;
    scalarfield rho;
    scalarfield alpha;

    // Atlas coords
    std::vector<std::vector<vector2field>> atlas_updates;
    std::vector<std::vector<vector2field>> grad_atlas_updates;
    std::vector<scalarfield> atlas_coords3;
    std::vector<vector2field> atlas_directions;
    std::vector<vector2field> atlas_residuals;
    std::vector<vector2field> atlas_residuals_last;
    std::vector<vector2field> atlas_q_vec;
};

template<>
inline void Method_Solver<Solver::LBFGS_Atlas>::Initialize()
{
    this->atlas_updates = std::vector<std::vector<vector2field>>(
        this->noi, std::vector<vector2field>( n_lbfgs_memory, vector2field( this->nos, { 0, 0 } ) ) );
    this->grad_atlas_updates = std::vector<std::vector<vector2field>>(
        this->noi, std::vector<vector2field>( n_lbfgs_memory, vector2field( this->nos, { 0, 0 } ) ) );
    this->rho                  = scalarfield( n_lbfgs_memory, 0 );
    this->alpha                = scalarfield( n_lbfgs_memory, 0 );
    this->forces               = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->forces_virtual       = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->atlas_coords3        = std::vector<scalarfield>( this->noi, scalarfield( this->nos, 1 ) );
    this->atlas_directions     = std::vector<vector2field>( this->noi, vector2field( this->nos, { 0, 0 } ) );
    this->atlas_residuals      = std::vector<vector2field>( this->noi, vector2field( this->nos, { 0, 0 } ) );
    this->atlas_residuals_last = std::vector<vector2field>( this->noi, vector2field( this->nos, { 0, 0 } ) );
    this->atlas_q_vec          = std::vector<vector2field>( this->noi, vector2field( this->nos, { 0, 0 } ) );
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

        Backend::transform(
            SPIRIT_PAR image.begin(), image.end(), forces[img].begin(), forces_virtual[img].begin(),
            [] SPIRIT_LAMBDA( const Vector3 & s, const Vector3 & f ) { return s.cross( f ); } );

        Solver_Kernels::atlas_calc_gradients( grad_ref, image, this->forces[img], this->atlas_coords3[img] );
    }

    // Calculate search direction
    Solver_Kernels::lbfgs_get_searchdir(
        this->local_iter, this->rho, this->alpha, this->atlas_q_vec, this->atlas_directions, this->atlas_updates,
        this->grad_atlas_updates, this->atlas_residuals, this->atlas_residuals_last, n_lbfgs_memory, maxmove );

    scalar a_norm_rms = 0;
    // Scale by averaging
    for( int img = 0; img < noi; img++ )
    {
        a_norm_rms = std::max(
            a_norm_rms,
            scalar( sqrt(
                Backend::transform_reduce(
                    SPIRIT_PAR this->atlas_directions[img].begin(), this->atlas_directions[img].end(), scalar( 0.0 ),
                    Backend::plus<scalar>{}, [] SPIRIT_LAMBDA( const Vector2 & v ) { return v.squaredNorm(); } )
                / nos ) ) );
    }
    scalar scaling = ( a_norm_rms > maxmove ) ? maxmove / a_norm_rms : 1.0;

    for( int img = 0; img < noi; img++ )
    {
        Backend::for_each(
            SPIRIT_PAR atlas_directions[img].begin(), atlas_directions[img].end(),
            [scaling] SPIRIT_LAMBDA( Vector2 & d ) { d *= scaling; } );

        // Rotate spins
        Solver_Kernels::atlas_rotate(
            *this->configurations[img], this->atlas_coords3[img], this->atlas_directions[img] );
    }

    const auto condition = [this]( const scalarfield & a3 )
    { return Solver_Kernels::ncg_atlas_check_coordinates( *this->configurations[0], a3, -0.6 ); };
    if( std::any_of( this->atlas_coords3.begin(), this->atlas_coords3.end(), condition ) )
    {
        const auto m_inverse = [] SPIRIT_LAMBDA( const scalar s ) { return scalar( 1.0 ) / s; };
        Backend::transform( SPIRIT_PAR rho.begin(), rho.end(), rho.begin(), m_inverse );

        for( int img = 0; img < noi; ++img )
            Solver_Kernels::lbfgs_atlas_transform_direction(
                *this->configurations[img], this->atlas_coords3[img], this->atlas_updates[img],
                this->grad_atlas_updates[img], this->atlas_directions[img], this->atlas_residuals_last[img],
                this->rho );

        Backend::transform( SPIRIT_PAR rho.begin(), rho.end(), rho.begin(), m_inverse );
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

} // namespace Spin

} // namespace Engine

#endif
