#pragma once
#ifndef SOLVER_KERNELS_H
#define SOLVER_KERNELS_H

#include <vector>
#include <memory>

#include <Eigen/Core>

#include <engine/Vectormath_Defines.hpp>

namespace Engine
{
    namespace Solver_Kernels
    {
        // Utility function for the SIB Solver
        void sib_transform(const vectorfield & spins, const vectorfield & force, vectorfield & out);

        // Full NCG Kernel for starters
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