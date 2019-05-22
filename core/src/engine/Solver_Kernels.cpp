#ifndef SPIRIT_USE_CUDA

#include <engine/Vectormath.hpp>
#include <engine/Manifoldmath.hpp>
#include <utility/Constants.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>

#include <Eigen/Dense>

#include <array>
#include <algorithm>

using namespace Utility;
using Utility::Constants::Pi;

namespace Engine
{
namespace Solver_Kernels
{
    void sib_transform(const vectorfield & spins, const vectorfield & force, vectorfield & out)
    {
        #pragma omp parallel for
        for (unsigned int i = 0; i < spins.size(); ++i)
        {
            Vector3 A = 0.5 * force[i];

            // 1/determinant(A)
            scalar detAi = 1.0 / (1 + A.squaredNorm());

            // calculate equation without the predictor?
            Vector3 a2 = spins[i] - spins[i].cross(A);

            out[i][0] = (a2[0] * (A[0] * A[0] + 1   ) + a2[1] * (A[0] * A[1] - A[2]) + a2[2] * (A[0] * A[2] + A[1])) * detAi;
            out[i][1] = (a2[0] * (A[1] * A[0] + A[2]) + a2[1] * (A[1] * A[1] + 1   ) + a2[2] * (A[1] * A[2] - A[0])) * detAi;
            out[i][2] = (a2[0] * (A[2] * A[0] - A[1]) + a2[1] * (A[2] * A[1] + A[0]) + a2[2] * (A[2] * A[2] + 1   )) * detAi;
        }
    }



    inline scalar inexact_line_search(scalar r, scalar E0, scalar Er, scalar g0, scalar gr)
    {
        scalar c1 = - 2*(Er - E0) / std::pow(r, 3) + (gr + g0) / std::pow(r, 2);
        scalar c2 = 3*(Er - E0) / std::pow(r, 2) - (gr + 2*g0) / r;
        scalar c3 = g0;
        scalar c4 = E0;

        return std::abs( (-c2 + std::sqrt(c2*c2 - 3*c1*c3)) / (3*c1) ) / r;
    }


    void full_inexact_line_search(const Data::Spin_System & system,
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

        scalar g0 = 0;
        scalar gr = 0;
        #pragma omp parallel for reduction(+:g0) reduction(+:gr)
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
        #pragma omp parallel for
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

    scalar ncg_beta_polak_ribiere(vectorfield & image, vectorfield & force, vectorfield & residual,
        vectorfield & residual_last, vectorfield & force_virtual)
    {
        scalar dt = 1e-3;
        // scalar dt = this->systems[0]->llg_parameters->dt;

        scalar top=0, bot=0;

        #pragma omp parallel for
        for( int i=0; i<image.size(); ++i )
        {
            // Set residuals
            residual_last[i] = residual[i];
            residual[i] = image[i].cross(force[i]);
            // TODO: this is for comparison with VP etc. and needs to be fixed!
            //       in fact, all solvers should use the force, not dt*force=displacement
            force_virtual[i] = dt * residual[i];

            bot += residual_last[i].dot(residual_last[i]);
            // Polak-Ribiere formula
            // TODO: this finite difference *should* be done covariantly (i.e. displaced)
            // Vectormath::rotate(residual_last[i], axis[i], step_size * angle[i], residual_last[i]);
            top += residual[i].dot( residual[i] - residual_last[i] );
            // Fletcher-Reeves formula
            // top += residual[i].dot( residual[i] );
        }
        if( std::abs(bot) > 0 )
            return std::max(top/bot, scalar(0));
        else
            return 0;
    }

    scalar ncg_dir_max(vectorfield & direction, vectorfield & residual, scalar beta, vectorfield & axis)
    {
        scalar dir_max = 0;
        #pragma omp parallel for reduction(max : dir_max)
        for( int i=0; i<direction.size(); ++i )
        {
            // direction = residual + beta*direction
            direction[i] = residual[i] + beta*direction[i];
            scalar dir_norm_i = direction[i].norm();
            // direction[i] = residual[i] + beta[img]*residual_last[i];
            axis[i] = direction[i].normalized();
            if( dir_norm_i > dir_max )
                dir_max = dir_norm_i;
            // dir_avg += dir_norm_i;
            // angle[i] = direction[i].norm();
        }
        return dir_max;
    }

    void ncg_rotate(vectorfield & direction, vectorfield & axis, scalarfield & angle, scalar normalization, const vectorfield & image, vectorfield & image_displaced)
    {
        #pragma omp parallel for
        for( int i=0; i<image.size(); ++i )
        {
            // Set rotation
            angle[i] = direction[i].norm() / normalization;
            // Rotate
            Vectormath::rotate(image[i], axis[i], angle[i], image_displaced[i]);
        }
    }

    void ncg_rotate_2(vectorfield & image, vectorfield & residual, vectorfield & axis, scalarfield & angle, scalar step_size)
    {
        #pragma omp parallel for
        for( int i=0; i<image.size(); ++i )
        {
            Vectormath::rotate(image[i], axis[i], step_size * angle[i], image[i]);
            Vectormath::rotate(residual[i], axis[i], step_size * angle[i], residual[i]);
        }
    }
}
}

#endif