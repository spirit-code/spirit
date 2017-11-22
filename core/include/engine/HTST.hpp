#pragma once
#ifndef HTST_H
#define HTST_H

#include "Spirit_Defines.h"
#include <data/Spin_System.hpp>
#include <engine/Vectormath_Defines.hpp>

namespace Engine
{
    namespace HTST
    {
        // Note the two images should correspond to one minimum and one saddle point
        scalar Get_Prefactor(std::shared_ptr<Data::Spin_System> system_minimum, std::shared_ptr<Data::Spin_System> system_sp);

        // Calculate the 'a' component of the prefactor
        void Calculate_a(const vectorfield & spins, const MatrixX & hessian,
            const MatrixX & basis, const MatrixX & eigenbasis, VectorX & a);

        // Calculate the Velocity matrix
        void Calculate_Velocity_Matrix(const vectorfield & spins, const MatrixX & hessian, MatrixX & velocity);

        // Prefactor Calculation from intermediate values (mode volume, eigenvalues, ...)
        void Calculate_Prefactor(int nos, int n_zero_modes_minimum, int n_zero_modes_sp, scalar volume_minimum, scalar volume_sp,
            const VectorX & eig_min, const VectorX & eig_sp, const VectorX & a, scalar & prefactor, scalar & exponent);
        
        // Calculate the zero volume of a spin system
        scalar Calculate_Zero_Volume(const std::shared_ptr<Data::Spin_System> system);

        // Returns the index of the first vector in vf which is approximately equal to vec
        int find_pos(const Vector3 & vec, const vectorfield & vf);

        // Generates the geodesic Hessian in 2N-representation and calculates it's eigenvalues and eigenvectors
        void Geodesic_Eigen_Decomposition(const vectorfield & image, const vectorfield & gradient, const MatrixX & hessian,
            MatrixX & hessian_geodesic_3N, MatrixX & hessian_geodesic_2N, VectorX & eigenvalues, MatrixX & eigenvectors);
    };
}


#endif