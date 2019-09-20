#pragma once
#ifndef SOLVER_LBFGS_ATLAS_HPP
#define SOLVER_LBFGS_ATLAS_HPP

#include <utility/Constants.hpp>
// #include <utility/Exception.hpp>
#include <algorithm>

using namespace Utility;

template <> inline
void Method_Solver<Solver::LBFGS_Atlas>::Initialize ()
{
    this->jmax    = 500;    // max iterations
    this->n       = 50;     // restart every n iterations XXX: what's the appropriate val?
    this->n_lbfgs_memory = 10; // how many updates the solver tracks to estimate the hessian
    this->n_updates = intfield( this->noi, 0 );

    this->atlas_updates      = std::vector<std::vector<vector2field>>( this->noi, std::vector<vector2field>( this->n_lbfgs_memory, vector2field(this->nos, { 0,0 } ) ));
    this->grad_atlas_updates = std::vector<std::vector<vector2field>>( this->noi, std::vector<vector2field>( this->n_lbfgs_memory, vector2field(this->nos, { 0,0 } ) ));
    this->rho_temp           = std::vector<scalarfield>( this->noi, scalarfield( this->n_lbfgs_memory, 0 ) );
    this->alpha_temp         = std::vector<scalarfield>( this->noi, scalarfield( this->n_lbfgs_memory, 0 ) );

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

template <> inline
void Method_Solver<Solver::LBFGS_Atlas>::Iteration()
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

        auto dot2D = [](const Vector2 & x, const Vector2 y) {return x.dot(y);};

        // Update virtual force
        for(int i=0; i<this->nos; ++i)
            this->forces_virtual[img][i] = image[i].cross(this->forces[img][i]);

        // Calculate residuals for current parameters and save the old residuals
        Solver_Kernels::ncg_atlas_residual(a_residuals, a_residuals_last, image, this->forces[img], a3_coords);

        // Keep track of residual updates
        if(this->iteration > 0)
        {
            int idx = (this->iteration-1) % n_lbfgs_memory;
            #pragma omp parallel for
            for(int i=0; i<this->nos; i++)
            {
                this->grad_atlas_updates[img][idx][i] = a_residuals[i] - a_residuals_last[i];
            }

            this->rho_temp[img][idx] = 1/Vectormath::reduce(this->grad_atlas_updates[img][idx], this->atlas_updates[img][idx], dot2D);
        }

        // Reset direction to steepest descent if line search failed
        if(!this->finish[img])
        {
            fmt::print("resetting\n");
            this->n_updates[img] = 0;
        }

        // Calculate new search direction
        Solver_Kernels::lbfgs_atlas_get_descent_direction(this->iteration, this->n_updates[img], a_directions, a_residuals, this->atlas_updates[img], this->grad_atlas_updates[img], this->rho_temp[img], this->alpha_temp[img]);
        this->a_direction_norm[img] = Vectormath::reduce(a_directions, a_directions, dot2D);
        // Manifoldmath::norm( a_directions );

        if(this->n_updates[img] < this->n_lbfgs_memory)
           this->n_updates[img]++;


        // Debug
        // for(int i=0; i<n_lbfgs_memory; i++)
        //     fmt::print("rho_temp[{}] = {}\n", i, this->rho_temp[img][i]);

        // for(int i=0; i<this->nos; i++)
        // {
        //     fmt::print("a_direction[{}] = [{}, {}, {}]\n", i, this->a_directions[img][i][0], this->a_directions[img][i][1], this->a_directions[img][i][2]);

        //     // if(std::isnan(this->a_directions[img][i][0]))
        //     //     throw 0;
        // }

        // Before the line search, set step_size and precalculate E0 and g0
        step_size[img] = 1.0;
        E0[img]        = this->systems[img]->hamiltonian->Energy(image);
        g0[img]        = 0;

        #pragma omp parallel for reduction(+:g0)
        for( int i=0; i<this->nos; ++i )
        {
            g0[img] -= a_residuals[i].dot(a_directions[i]) / a_direction_norm[img];
            this->finish[img] = false;
        }
    }

    int n_search_steps     = 0;
    int n_search_steps_max = 20;
    bool run = true;;
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
            int idx = this->iteration % this->n_lbfgs_memory;
            if(this->finish[img]) // only if line search was successfull
            {
                this->atlas_updates[img][idx][i] = a_coords_displaced[i] - a_coords[i]; // keep track of atlas_updates
                a_coords[i] = a_coords_displaced[i];
                image[i]    = image_displaced[i];
            } else {
                this->atlas_updates[img][idx][i] = {0,0}; // keep track of atlas_updates
            }
        }

        Solver_Kernels::ncg_atlas_check_coordinates(image, a_coords, a3_coords, this->atlas_directions[img]);

    }
}

template <> inline
std::string Method_Solver<Solver::LBFGS_Atlas>::SolverName()
{
    return "LBFGS_Atlas";
}

template <> inline
std::string Method_Solver<Solver::LBFGS_Atlas>::SolverFullName()
{
    return "Limited memory Broyden-Fletcher-Goldfarb-Shanno";
}

#endif