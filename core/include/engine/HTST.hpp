#pragma once
#ifndef SPIRIT_CORE_ENGINE_HTST_HPP
#define SPIRIT_CORE_ENGINE_HTST_HPP
#ifndef SPIRIT_SKIP_HTST

#include "Spirit_Defines.h"
#include <data/Spin_System.hpp>
#include <data/Spin_System_Chain.hpp>
#include <engine/Vectormath_Defines.hpp>

namespace Engine
{
namespace HTST
{

// Note the two images should correspond to one minimum and one saddle point
void Calculate( Data::HTST_Info & htst_info, int n_eigenmodes_keep = 0 );

// Calculate the 'a' component of the prefactor
void Calculate_Perpendicular_Velocity(
    const vectorfield & spins, const scalarfield & mu_s, const MatrixX & hessian, const MatrixX & basis,
    const MatrixX & eigenbasis, VectorX & a );

// Calculate the Velocity matrix
void Calculate_Dynamical_Matrix(
    const vectorfield & spins, const scalarfield & mu_s, const MatrixX & hessian, MatrixX & velocity );

// Calculate the zero volume of a spin system
scalar Calculate_Zero_Volume( const std::shared_ptr<Data::Spin_System> system );

// Generates the geodesic Hessian in 2N-representation and calculates it's eigenvalues and eigenvectors
void Geodesic_Eigen_Decomposition(
    const vectorfield & image, const vectorfield & gradient, const MatrixX & hessian, MatrixX & hessian_geodesic_3N,
    MatrixX & hessian_geodesic_2N, VectorX & eigenvalues, MatrixX & eigenvectors );

scalar Calculate_Zero_Volume( const std::shared_ptr<Data::Spin_System> system );

} // namespace HTST
} // namespace Engine

#endif
#endif