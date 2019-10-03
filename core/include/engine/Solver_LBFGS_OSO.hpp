#pragma once
#ifndef SOLVER_LBFGS_OSO_HPP
#define SOLVER_LBFGS_OSO_HPP

#include <utility/Constants.hpp>
// #include <utility/Exception.hpp>
#include <algorithm>

using namespace Utility;

template <> inline
void Method_Solver<Solver::LBFGS_OSO>::Initialize ()
{
    this->jmax = 500;    // max iterations
    this->n    = 50;     // restart every n iterations XXX: what's the appropriate val?
    this->n_lbfgs_memory = 5; // how many updates the solver tracks to estimate the hessian
    this->n_updates = intfield( this->noi, 0 );

    this->a_updates    = std::vector<std::vector<vectorfield>>( this->noi, std::vector<vectorfield>( this->n_lbfgs_memory, vectorfield(this->nos, { 0,0,0 } ) ));
    this->grad_updates = std::vector<std::vector<vectorfield>>( this->noi, std::vector<vectorfield>( this->n_lbfgs_memory, vectorfield(this->nos, { 0,0,0 } ) ));
    this->rho_temp     = std::vector<scalarfield>( this->noi, scalarfield( this->n_lbfgs_memory, 0 ) );
    this->alpha_temp   = std::vector<scalarfield>( this->noi, scalarfield( this->n_lbfgs_memory, 0 ) );

    this->forces                   = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0,0,0 } ) );
    this->forces_virtual           = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0,0,0 } ) );
    this->a_directions             = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0,0,0 } ) );
    this->a_residuals              = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0,0,0 } ) );
    this->a_residuals_last         = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0,0,0 } ) );
    this->step_size                = scalarfield(this->noi, 0);
};

/*
    Implemented according to Aleksei Ivanov's paper: https://arxiv.org/abs/1904.02669
    TODO: reference painless conjugate gradients
    See also Jorge Nocedal and Stephen J. Wright 'Numerical Optimization' Second Edition, 2006 (p. 121).
*/

template <> inline
void Method_Solver<Solver::LBFGS_OSO>::Iteration()
{
    scalar max_rot = Constants::Pi * 0.2;

    // Current force
    this->Calculate_Force( this->configurations, this->forces );

    #pragma omp parallel for
    for( int img=0; img<this->noi; img++ )
    {
        auto& image            = *this->configurations[img];
        auto& a_coords         = this->a_coords[img];
        auto& a_directions     = this->a_directions[img];
        auto& a_residuals      = this->a_residuals[img];
        auto& a_residuals_last = this->a_residuals_last[img];

        // Update virtual force
        for(int i=0; i<this->nos; ++i)
            this->forces_virtual[img][i] = image[i].cross(this->forces[img][i]);

        // Calculate residuals for current parameters and save the old residuals
        Solver_Kernels::ncg_OSO_residual(a_residuals, a_residuals_last, image, this->forces[img]);

        // Keep track of residual updates
        if(this->iteration > 0)
        {
            int idx = (this->iteration-1) % n_lbfgs_memory;
            #pragma omp parallel for
            for(int i=0; i<this->nos; i++)
            {
                this->grad_updates[img][idx][i] = a_residuals[i] - a_residuals_last[i];
            }

            this->rho_temp[img][idx] = 1/Vectormath::dot(this->grad_updates[img][idx], this->a_updates[img][idx]);

            if(this->rho_temp[img][idx] > 0)
            {
                this->n_updates[img] = 0;
            }

        }

        // Calculate new search direction
        Solver_Kernels::lbfgs_get_descent_direction(this->iteration, this->n_updates[img], a_directions, a_residuals, this->a_updates[img], this->grad_updates[img], this->rho_temp[img], this->alpha_temp[img]);

        if(this->n_updates[img] < this->n_lbfgs_memory)
           this->n_updates[img]++;

        step_size[img] = 1.0;
    }

    Solver_Kernels::ncg_OSO_displace( this->configurations, this->a_directions, this->step_size, max_rot );

    for (int img=0; img<this->noi; img++)
    {
        auto& image = *this->configurations[img];

        // Update current image
        for(int i=0; i<image.size(); i++)
        {
            int idx = this->iteration % this->n_lbfgs_memory;
            this->a_updates[img][idx][i] = step_size[img] * a_directions[img][i]; // keep track of a_updates
        }
    }
}

template <> inline
std::string Method_Solver<Solver::LBFGS_OSO>::SolverName()
{
    return "LBFGS_OSO";
}

template <> inline
std::string Method_Solver<Solver::LBFGS_OSO>::SolverFullName()
{
    return "Limited memory Broyden-Fletcher-Goldfarb-Shanno with exponential transform";
}

#endif