#pragma once
#ifndef SPIRIT_CORE_ENGINE_SPIN_HTST_HPP
#define SPIRIT_CORE_ENGINE_SPIN_HTST_HPP
#ifndef SPIRIT_SKIP_HTST

#include <Spirit/Spirit_Defines.h>
#include <data/State.hpp>
#include <engine/Vectormath_Defines.hpp>

namespace Engine
{

namespace Spin
{

namespace HTST
{

// Note the two images should correspond to one minimum and one saddle point
template<typename system_t>
void Calculate( Data::HTST_Info<system_t> & htst_info, int n_eigenmodes_keep = 0 );

// Calculate the 'a' component of the prefactor
void Calculate_Perpendicular_Velocity(
    const vectorfield & spins, const scalarfield & mu_s, const MatrixX & hessian, const MatrixX & basis,
    const MatrixX & eigenbasis, VectorX & a );

// Calculate the Velocity matrix
void Calculate_Dynamical_Matrix(
    const vectorfield & spins, const scalarfield & mu_s, const MatrixX & hessian, MatrixX & velocity );

// Calculate the zero volume of a spin system
scalar Calculate_Zero_Volume( const system_t & system );

// Generates the geodesic Hessian in 2N-representation and calculates it's eigenvalues and eigenvectors
void Geodesic_Eigen_Decomposition(
    const vectorfield & image, const vectorfield & gradient, const MatrixX & hessian, MatrixX & hessian_geodesic_3N,
    MatrixX & hessian_geodesic_2N, VectorX & eigenvalues, MatrixX & eigenvectors );

} // namespace HTST

} // namespace Spin

} // namespace Engine

#endif
#endif
