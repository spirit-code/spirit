#pragma once

#include <utility/Constants.hpp>

#include <algorithm>

using namespace Utility;

template <> inline
void Method_Solver<Solver::NCG>::Initialize ()
{
    this->jmax    = 500;    // max iterations
    this->n       = 50;     // restart every n iterations XXX: what's the appropriate val?

    // Polak-Ribiere criterion
    this->beta  = scalarfield( this->noi, 0 );

    this->forces  = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->forces_virtual  = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->forces_displaced  = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );

    this->residuals  = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->residuals_last  = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );

    this->directions = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->directions_displaced = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );

    this->axes  = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->angles  = std::vector<scalarfield>( this->noi, scalarfield( this->nos, 0 ) );

    this->configurations_displaced = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for (int i=0; i<this->noi; i++)
        configurations_displaced[i] = std::shared_ptr<vectorfield>( new vectorfield( this->nos, {0, 0, 0} ) );
};



/*
    Implemented according to Aleksei Ivanov's paper: https://arxiv.org/abs/1904.02669
    TODO: reference painless conjugate gradients
    See also Jorge Nocedal and Stephen J. Wright 'Numerical Optimization' Second Edition, 2006 (p. 121)

    Template instantiation of the Simulation class for use with the NCG Solver
    The method of nonlinear conjugate gradients is a proven and effective solver.
*/
template <> inline
void Method_Solver<Solver::NCG>::Iteration ()
{
    // Current force
    this->Calculate_Force( this->configurations, this->forces );

    scalar dir_norm = 0;
    scalar dir_max = 0;
    // scalar dir_avg = 0;
    scalar top=0, bot=0;
    for (int img=0; img<this->noi; img++)
    {
        auto& image         = *this->configurations[img];
        auto& image_displaced = *this->configurations_displaced[img];
        auto& force         = this->forces[img];
        auto& force_virtual = this->forces_virtual[img];
        auto& residual      = this->residuals[img];
        auto& residual_last = this->residuals_last[img];
        auto& direction     = this->directions[img];
        auto& direction_displaced = this->directions_displaced[img];
        auto& angle         = this->angles[img];
        auto& axis          = this->axes[img];

        scalar dt = this->systems[0]->llg_parameters->dt;
        this->beta[img] = Solver_Kernels::ncg_beta_polak_ribiere(image, force, residual, residual_last, force_virtual);
        scalar dir_max = Solver_Kernels::ncg_dir_max(direction, residual, this->beta[img], axis);

        // dir_avg /= image.size();
        if( dir_max < 1e-12 )
            return;

        Solver_Kernels::ncg_rotate(direction, axis, angle, dir_max, image, image_displaced);
    }

    // Displaced force
    this->Calculate_Force( this->configurations_displaced, this->forces_displaced );

    for (int img=0; img<this->noi; img++)
    {
        auto& image             = *this->configurations[img];
        auto& image_displaced   = *this->configurations_displaced[img];
        auto& force_displaced   = this->forces_displaced[img];
        auto& residual_last     = this->residuals_last[img];
        auto& residual          = this->residuals[img];
        auto& direction         = this->directions[img];
        // auto& direction_displaced = this->directions_displaced[img];
        auto& angle             = this->angles[img];
        auto& axis              = this->axes[img];

        scalar step_size = 1.0;
        int n_step = 0;
        Solver_Kernels::full_inexact_line_search(*this->systems[img], image, image_displaced,
            residual, force_displaced, angle, axis, step_size, n_step);

        Solver_Kernels::ncg_rotate_2(image, residual, axis, angle, step_size);

        // For debugging
        // scalar angle_norm = Vectormath::sum(angle);
        // std::cerr << fmt::format("dir_max = {:^14}  dir_avg = {:^14}    beta = {:^14}  =  {:^14} / {:^14}   angle = {:^14}\n",
        //     dir_max, dir_avg, this->beta[img], top, bot, angle_norm);
    }
}

template <> inline
std::string Method_Solver<Solver::NCG>::SolverName()
{
    return "NCG";
}

template <> inline
std::string Method_Solver<Solver::NCG>::SolverFullName()
{
    return "Nonlinear conjugate gradients";
}