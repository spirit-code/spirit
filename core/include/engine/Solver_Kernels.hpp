#pragma once
#ifndef SOLVER_KERNELS_H
#define SOLVER_KERNELS_H

#include <vector>
#include <memory>

#include <Eigen/Core>
#include <complex>
#include <engine/Vectormath_Defines.hpp>
#include <data/Spin_System.hpp>

namespace Engine
{
namespace Solver_Kernels
{
    // SIB
    void sib_transform(const vectorfield & spins, const vectorfield & force, vectorfield & out);

    // NCG
    inline scalar inexact_line_search(scalar r, scalar E0, scalar Er, scalar g0, scalar gr)
    {
        scalar c1 = -2*(Er - E0) / std::pow(r, 3) + (gr + g0) / std::pow(r, 2);
        scalar c2 = 3*(Er - E0) / std::pow(r, 2) - (gr + 2*g0) / r;
        scalar c3 = g0;
        // scalar c4 = E0;
        return std::abs( (-c2 + std::sqrt(c2*c2 - 3*c1*c3)) / (3*c1) ) / r;
    }
    scalar ncg_beta_polak_ribiere(vectorfield & image, vectorfield & force, vectorfield & residual,
        vectorfield & residual_last, vectorfield & force_virtual);
    scalar ncg_dir_max(vectorfield & direction, vectorfield & residual, scalar beta, vectorfield & axis);
    void full_inexact_line_search(const Data::Spin_System & system,
        const vectorfield & image, vectorfield & image_displaced,
        const vectorfield & force, const vectorfield & force_displaced,
        const scalarfield & angle, const vectorfield & axis, scalar & step_size, int & n_step);
    void ncg_rotate(vectorfield & direction, vectorfield & axis, scalarfield & angle,
        scalar normalization, const vectorfield & image, vectorfield & image_displaced);
    void ncg_rotate_2(vectorfield & image, vectorfield & residual, vectorfield & axis,
        scalarfield & angle, scalar step_size);

    void ncg_OSO_residual( vectorfield & residuals, vectorfield & residuals_last, const vectorfield & spins, const vectorfield & a_coords,
                          const vectorfield & forces, bool approx = true);
    void ncg_OSO_a_to_spins(vectorfield & spins, const vectorfield & a_coords, const vectorfield & reference_spins);

    void ncg_OSO_update_reference_spins(vectorfield & reference_spins, vectorfield & a_coords, const vectorfield & spins);

    void ncg_OSO_displace( std::vector<std::shared_ptr<vectorfield>> & configurations_displaced, std::vector<std::shared_ptr<vectorfield>> & reference_configurations, std::vector<vectorfield> & a_coords,
                           std::vector<vectorfield> & a_coords_displaced, std::vector<vectorfield> & a_directions, std::vector<bool> finish, scalarfield step_size );

    void ncg_OSO_line_search( std::vector<std::shared_ptr<vectorfield>> & configurations_displaced, std::vector<vectorfield> & a_coords_displaced, std::vector<vectorfield> & a_directions,
                              std::vector<vectorfield> & forces_displaced, std::vector<vectorfield> & a_residuals_displaced, std::vector<std::shared_ptr<Data::Spin_System>> systems, std::vector<bool> & finish,
                              scalarfield & E0, scalarfield & g0, scalarfield & a_direction_norm, scalarfield & step_size  );

    inline scalar ncg_OSO_dir_max(vectorfield & a_directions)
    {
        scalar res = 0;
        # pragma omp parallel for
        for(int i=0; i<a_directions.size(); i++)
        {
            if( a_directions[i].norm() > res )
            {
                res = a_directions[i].norm();
            }
        }
        return res;
    }

    inline bool ncg_OSO_wolfe_conditions(scalar E0, scalar Er, scalar g0, scalar gr, scalar step)
    {
        constexpr scalar c1 = 1e-4;
        constexpr scalar c2 = 0.9;
        return (Er <= E0 + c1 * step * gr) && (std::abs(gr) <= c2 * std::abs(g0) );
    }

    void lbfgs_get_descent_direction(int iteration, int n_lbfgs_memory, vectorfield & a_direction ,vectorfield & residual, const std::vector<vectorfield> & spin_updates, const std::vector<vectorfield> & grad_updates, const scalarfield & rho_temp, scalarfield & alpha_temp);
}
}

#endif