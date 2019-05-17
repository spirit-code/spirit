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


inline scalar inexact_line_search(scalar r, scalar E0, scalar Er, scalar g0, scalar gr)
{
    scalar c1 = - 2*(Er - E0) / std::pow(r, 3) + (gr + g0) / std::pow(r, 2);
    scalar c2 = 3*(Er - E0) / std::pow(r, 2) - (gr + 2*g0) / r;
    scalar c3 = g0;
    scalar c4 = E0;

    return std::abs( (-c2 + std::sqrt(c2*c2 - 3*c1*c3)) / (3*c1) ) / r;
}


inline void full_inexact_line_search(const Data::Spin_System & system,
    const vectorfield & image, vectorfield & image_displaced,
    const vectorfield & force, const vectorfield & force_displaced,
    const scalarfield & angle, const vectorfield & axis, scalar & step_size, int & n_step)
{
    // Calculate geodesic distance between image and image_displaced, if not pre-determined
    scalar r = Manifoldmath::dist_geodesic(image, image_displaced);
    if( r < 1e-6 )
    {
        step_size = 0;
        return;
    }

    scalar E0 = system.hamiltonian->Energy(image);
    // E0 = this->systems[img]->E;
    scalar Er = system.hamiltonian->Energy(image_displaced);

    // TODO: parallelize reduction
    scalar g0 = 0;
    scalar gr = 0;
    for( int i=0; i<image.size(); ++i )
    {
        g0 += force[i].dot(axis[i]);
        // TODO: displace dir by rotating into other spin
        // ACTUALLY: the direction is orthogonal to the rotation plane, so it does not change
        gr += ( image_displaced[i].cross(force_displaced[i]) ).dot(axis[i]);
    }

    // Approximate ine search
    ++n_step;
    step_size *= inexact_line_search(r, E0, Er, g0, gr);// * Constants::gamma / Constants::mu_B;
    for( int i=0; i<image.size(); ++i )
    {
        Vectormath::rotate(image[i], axis[i], step_size * angle[i], image_displaced[i]);
    }
    Er = system.hamiltonian->Energy(image_displaced);
    // this->Calculate_Force( this->configurations_displaced, this->forces_displaced );
    if( n_step < 20 && Er > E0+std::abs(E0)*1e-4 )
    {
        full_inexact_line_search(system, image, image_displaced, force, force_displaced, angle, axis, step_size, n_step);
    }
}

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
        auto& residual      = this->residuals[img];
        auto& residual_last = this->residuals_last[img];
        auto& direction     = this->directions[img];
        auto& direction_displaced = this->directions_displaced[img];
        auto& angle         = this->angles[img];
        auto& axis          = this->axes[img];

        // TODO: parallelize
        for( int i=0; i<image.size(); ++i )
        {
            // Set residuals
            residual_last[i] = residual[i];
            residual[i] = image[i].cross(force[i]);
            // TODO: this is for comparison with VP etc. and needs to be fixed!
            //       in fact, all solvers should use the force, not dt*force=displacement
            this->forces_virtual[img][i] = this->systems[0]->llg_parameters->dt * residual[i];

            bot += residual_last[i].dot(residual_last[i]);
            // Polak-Ribiere formula
            // TODO: this finite difference *should* be done covariantly (i.e. displaced)
            // Vectormath::rotate(residual_last[i], axis[i], step_size * angle[i], residual_last[i]);
            top += residual[i].dot( residual[i] - residual_last[i] );
            // Fletcher-Reeves formula
            // top += residual[i].dot( residual[i] );
        }
        if( std::abs(bot) > 0 )
            // Polak-Ribiere
            this->beta[img] = std::max(top/bot, scalar(0));
        else
            this->beta[img] = 0;
        for( int i=0; i<image.size(); ++i )
        {
            // direction = residual + beta*direction
            direction[i] = residual[i] + beta[img]*direction[i];
            scalar dir_norm_i = direction[i].norm();
            // direction[i] = residual[i] + beta[img]*residual_last[i];
            axis[i] = direction[i].normalized();
            if( dir_norm_i > dir_max )
                dir_max = direction[i].norm();
            // dir_avg += dir_norm_i;
            // angle[i] = direction[i].norm();
        }
        // dir_avg /= image.size();
        if( dir_max < 1e-12 )
            return;
        // TODO: parallelize
        for( int i=0; i<image.size(); ++i )
        {
            // Set rotation
            angle[i] = direction[i].norm() / dir_max;
            // Rotate
            Vectormath::rotate(image[i], axis[i], angle[i], image_displaced[i]);
        }
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
        full_inexact_line_search(*this->systems[img], image, image_displaced,
            residual, force_displaced, angle, axis, step_size, n_step);

        // TODO: parallelize
        for( int i=0; i<image.size(); ++i )
        {
            Vectormath::rotate(image[i], axis[i], step_size * angle[i], image[i]);
            Vectormath::rotate(residual[i], axis[i], step_size * angle[i], residual[i]);
        }
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