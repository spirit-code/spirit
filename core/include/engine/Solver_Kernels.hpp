#pragma once
#ifndef SOLVER_KERNELS_H
#define SOLVER_KERNELS_H

#include <vector>
#include <memory>

#include <Eigen/Core>

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
        scalar c1 = - 2*(Er - E0) / std::pow(r, 3) + (gr + g0) / std::pow(r, 2);
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
}
}

#endif