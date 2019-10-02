#pragma once
#ifndef SOLVER_NCG_ATLAS_HPP
#define SOLVER_NCG_ATLAS_HPP

#include <utility/Constants.hpp>

#include <algorithm>

using namespace Utility;

template <> inline
void Method_Solver<Solver::NCG_Atlas>::Initialize ()
{
    this->jmax    = 500;    // max iterations
    this->n       = 50;     // restart every n iterations XXX: what's the appropriate val?

    // Polak-Ribiere criterion
    this->beta  = scalarfield( this->noi, 0 );

    this->forces                    = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0,0,0 } ) );
    this->forces_virtual            = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0,0,0 } ) );
    this->forces_displaced          = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0,0,0 } ) );
    this->atlas_coords              = std::vector<vector2field>( this->noi, vector2field( this->nos, { 0,0 } ) );
    this->atlas_coords3             = std::vector< scalarfield >( this->noi, scalarfield(this->nos, 1) );
    this->atlas_coords_displaced    = std::vector<vector2field>( this->noi, vector2field( this->nos, { 0,0 } ) );
    this->atlas_directions          = std::vector<vector2field>( this->noi, vector2field( this->nos, { 0,0 } ) );
    this->a_direction_norm          = scalarfield     ( this->noi, 0 );
    this->atlas_residuals           = std::vector<vector2field>( this->noi, vector2field( this->nos, { 0,0 } ) );
    this->atlas_residuals_last      = std::vector<vector2field>( this->noi, vector2field( this->nos, { 0,0 } ) );
    this->atlas_residuals_displaced = std::vector<vector2field>( this->noi, vector2field( this->nos, { 0,0 } ) );
    this->configurations_displaced  = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    this->E0                        = scalarfield(this->noi, 0);
    this->g0                        = scalarfield(this->noi, 0);
    this->finish                    = std::vector<bool>( this->noi, false );
    this->step_size                 = scalarfield(this->noi, 0);
    for (int i=0; i<this->noi; i++)
    {
        configurations_displaced[i] = std::shared_ptr<vectorfield>( new vectorfield( this->nos, {0, 0, 0} ) );
        Solver_Kernels::ncg_spins_to_atlas( *this->configurations[i], this->atlas_coords[i], this->atlas_coords3[i] );
    }
};

/*
    Implemented according to an idea of F. Rybakov
    TODO: reference painless conjugate gradients
    See also Jorge Nocedal and Stephen J. Wright 'Numerical Optimization' Second Edition, 2006 (p. 121)

    Template instantiation of the Simulation class for use with the NCG Solver
    The method of nonlinear conjugate gradients is a proven and effective solver.
*/

template <> inline
void Method_Solver<Solver::NCG_Atlas>::Iteration()
{
    // Current force
    this->Calculate_Force( this->configurations, this->forces );

    #pragma omp parallel for
    for( int img=0; img<this->noi; img++ )
    {
        auto& image                 = *this->configurations[img];
        auto& image_displaced       = *this->configurations_displaced[img];
        auto& beta                  = this->beta[img];
        auto& a_coords              = this->atlas_coords[img];
        auto& a3_coords             = this->atlas_coords3[img];
        auto& a_coords_displaced    = this->atlas_coords_displaced[img];
        auto& a_directions          = this->atlas_directions[img];
        auto& a_residuals           = this->atlas_residuals[img];
        auto& a_residuals_last      = this->atlas_residuals_last[img];
        auto& a_residuals_displaced = this->atlas_residuals_displaced[img];

        // Update virtual force
        for(int i=0; i<this->nos; ++i)
            this->forces_virtual[img][i] = image[i].cross(this->forces[img][i]);

        // Calculate residuals for current parameters and save the old residuals
        Solver_Kernels::ncg_atlas_residual(a_residuals, a_residuals_last, image, this->forces[img], a3_coords);

        // Calculate beta
        scalar top = 0, bot = 0;
        for(int i=0; i<this->nos; i++)
        {
            top += a_residuals[i].dot(a_residuals[i] - a_residuals_last[i]);
            bot += a_residuals_last[i].dot(a_residuals_last[i]);
        }
        if( std::abs(bot) > 0 ) {
            this->beta[img] = std::max(top/bot, scalar(0));
        } else {
            this->beta[img] = 0;
        }

        // Reset direction to steepest descent if line search failed
        if(!this->finish[img])
            this->beta[img] = 0;

        // Calculate new search direction
        #pragma omp parallel for
        for(int i=0; i<this->nos; i++)
        {
            a_directions[i] *= beta;
            a_directions[i] += a_residuals[i];
        }

        this->a_direction_norm[img] = Solver_Kernels::ncg_atlas_norm( a_directions );
        // Manifoldmath::normalize( a_directions );

        // Before the line search, set step_size and precalculate E0 and g0
        step_size[img] = 1.0e0;
        E0[img]        = this->systems[img]->hamiltonian->Energy(image);
        g0[img]        = 0;

        #pragma omp parallel for reduction(+:g0[img])
        for( int i=0; i<this->nos; ++i )
        {
            g0[img] -= a_residuals[i].dot(a_directions[i]) / a_direction_norm[img];
            this->finish[img] = false;
        }
    }

    int n_search_steps     = 0;
    int n_search_steps_max = 20;
    bool run = true;
    while(run)
    {
        Solver_Kernels::ncg_atlas_displace(this->configurations_displaced, this->atlas_coords, this->atlas_coords3,
                                        this->atlas_coords_displaced, this->atlas_directions, this->finish, this->step_size);

        // Calculate forces for displaced spin directions
        this->Calculate_Force( configurations_displaced, forces_displaced );

        Solver_Kernels::ncg_atlas_line_search( this->configurations_displaced, this->atlas_coords_displaced, this->atlas_coords3, this->atlas_directions,
                             this->forces_displaced, this->atlas_residuals_displaced, this->systems, this->finish, E0, g0, this->a_direction_norm, step_size );

        n_search_steps++;
        fmt::print("line search step {}\n", n_search_steps);

        run = (n_search_steps >= n_search_steps_max) ? false : true;
        for(int img=0; img<this->noi; img++)
            run = (run && !finish[img]);
    }

    for (int img=0; img<this->noi; img++)
    {
        auto& image              = *this->configurations[img];
        auto& image_displaced    = *this->configurations_displaced[img];
        auto& a_coords           = this->atlas_coords[img];
        auto& a3_coords          = this->atlas_coords3[img];
        auto& a_coords_displaced = this->atlas_coords_displaced[img];

        // Update current image
        for(int i=0; i<image.size(); i++)
        {
            if(this->finish[img]) // only if line search was successfull
            {
                a_coords[i] = a_coords_displaced[i];
                image[i]    = image_displaced[i];
            }
        }
        Solver_Kernels::ncg_atlas_check_coordinates(image, a_coords, a3_coords, this->atlas_directions[img]);
    }
}

template <> inline
std::string Method_Solver<Solver::NCG_Atlas>::SolverName()
{
    return "NCG ATLAS";
}

template <> inline
std::string Method_Solver<Solver::NCG_Atlas>::SolverFullName()
{
    return "Nonlinear conjugate gradients with stereographic atlas";
}

#endif