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



    // NCG_OSO
    void ncg_OSO_residual( vectorfield & residuals, vectorfield & residuals_last, const vectorfield & spins, const vectorfield & forces, bool approx = true);
    void ncg_OSO_a_to_spins(vectorfield & spins, const vectorfield & a_coords, const vectorfield & reference_spins);

    void ncg_OSO_update_reference_spins(vectorfield & reference_spins, vectorfield & a_coords, const vectorfield & spins);

    // void ncg_OSO_displace( std::vector<std::shared_ptr<vectorfield>> & configurations_displaced, std::vector<std::shared_ptr<vectorfield>> & reference_configurations, std::vector<vectorfield> & a_coords,
                        //    std::vector<vectorfield> & a_coords_displaced, std::vector<vectorfield> & a_directions, std::vector<bool> finish, scalarfield step_size );

    void ncg_OSO_displace( std::vector<std::shared_ptr<vectorfield>> & configurations, std::vector<vectorfield> & a_directions, scalarfield & step_size, scalar max_rot);


    void ncg_OSO_line_search( std::vector<std::shared_ptr<vectorfield>> & configurations_displaced, std::vector<vectorfield> & a_coords_displaced, std::vector<vectorfield> & a_directions,
                              std::vector<vectorfield> & forces_displaced, std::vector<vectorfield> & a_residuals_displaced, std::vector<std::shared_ptr<Data::Spin_System>> systems, std::vector<bool> & finish,
                              scalarfield & E0, scalarfield & g0, scalarfield & a_direction_norm, scalarfield & step_size  );

    inline bool ncg_OSO_wolfe_conditions(scalar E0, scalar Er, scalar g0, scalar gr, scalar step)
    {
        constexpr scalar c1 = 1e-4;
        constexpr scalar c2 = 0.9;
        return (Er <= E0 + c1 * step * gr) && (std::abs(gr) <= c2 * std::abs(g0) );
    }


    // NCG_Atlas
    void ncg_atlas_residual( vector2field & residuals, vector2field & residuals_last, const vectorfield & spins,
                             const vectorfield & forces, const scalarfield & a3_coords );
    void ncg_atlas_to_spins(const vector2field & a_coords, const scalarfield & a3_coords, vectorfield & spins);
    void ncg_spins_to_atlas(const vectorfield & spins, vector2field & a_coords, scalarfield & a3_coords);

    bool ncg_atlas_check_coordinates(const vectorfield & spins, scalarfield & a3_coords);
    void ncg_atlas_transform_direction(const vectorfield & spins, vector2field & a_coords, scalarfield & a3_coords, vector2field & a_directions);

    void ncg_atlas_displace( std::vector<std::shared_ptr<vectorfield>> & configurations_displaced, std::vector<vector2field> & a_coords, std::vector<scalarfield> & a3_coords,
                             std::vector<vector2field> & a_coords_displaced, std::vector<vector2field> & a_directions, std::vector<bool> finish, scalarfield step_size );

    void ncg_atlas_line_search(  std::vector<std::shared_ptr<vectorfield>> & configurations_displaced, std::vector<vector2field> & a_coords_displaced, std::vector<scalarfield>  & a3_coords, std::vector<vector2field> & a_directions,
                                 std::vector<vectorfield> & forces_displaced, std::vector<vector2field> & a_residuals_displaced, std::vector<std::shared_ptr<Data::Spin_System>> systems, std::vector<bool> & finish,
                                 scalarfield & E0, scalarfield & g0, scalarfield & a_direction_norm, scalarfield & step_size);


    scalar ncg_atlas_norm(const vector2field & a_coords);

    scalar ncg_atlas_distance(const vector2field & a_coords1, const vector2field & a_coords2);

    // LBFGS_OSO
    void lbfgs_get_descent_direction(int iteration, field<int> & n_updates, field<vectorfield> & a_direction, const field<vectorfield> & residual, const field<field<vectorfield>> & a_updates, const field<field<vectorfield>> & grad_updates, const field<scalarfield> & rho_temp, field<scalarfield> & alpha_temp);

    // LBFGS_Atlas
    void lbfgs_atlas_get_descent_direction(int iteration, int n_updates, vector2field & a_direction, const vector2field & residual, const std::vector<vector2field> & a_updates, const std::vector<vector2field> & grad_updates, const scalarfield & rho_temp, scalarfield & alpha_temp);
    void lbfgs_atlas_transform_direction(const vectorfield & spins, vector2field & a_coords, scalarfield & a3_coords, field<vector2field> & directions);

}
}

#endif