#pragma once
#ifndef SPIRIT_CORE_ENGINE_SPIN_EIGENMODES_HPP
#define SPIRIT_CORE_ENGINE_SPIN_EIGENMODES_HPP

#include <data/Geometry.hpp>
#include <data/State.hpp>
#include <utility/Logging.hpp>

#include <memory>
#include <vector>

namespace Engine
{

namespace Spin
{

namespace Eigenmodes
{

// Check whether system members and EMA parameters are consistent with eachother
void Check_Eigenmode_Parameters( system_t & system );

// Calculate a systems eigenmodes according to its EMA parameters
void Calculate_Eigenmodes( system_t & system, int idx_img, int idx_chain );

// Calculate the full eigenspectrum of a Hessian (needs to be self-adjoint)
// gradient and hessian should be the 3N-dimensional representations without constraints
bool Hessian_Full_Spectrum(
    const Data::Geometry & geometry, const vectorfield & spins, const vectorfield & gradient, const MatrixX & hessian,
    MatrixX & tangent_basis, MatrixX & hessian_constrained, VectorX & eigenvalues, MatrixX & eigenvectors );

// Calculate a partial eigenspectrum of a Hessian
// gradient and hessian should be the 3N-dimensional representations without constraints
bool Hessian_Partial_Spectrum(
    const Data::Geometry & geometry, const vectorfield & spins, const vectorfield & gradient, const MatrixX & hessian,
    std::size_t n_modes, MatrixX & tangent_basis, MatrixX & hessian_constrained, VectorX & eigenvalues,
    MatrixX & eigenvcetors );

// Calculate a partial eigenspectrum of a sparse Hessian
// gradient and hessian should be the 3N-dimensional representations without constraints
bool Sparse_Hessian_Partial_Spectrum(
    const Data::Geometry & geometry, const vectorfield & spins, const vectorfield & gradient, const SpMatrixX & hessian,
    int n_modes, SpMatrixX & tangent_basis, SpMatrixX & hessian_constrained, VectorX & eigenvalues,
    MatrixX & eigenvectors );

} // end namespace Eigenmodes

} // namespace Spin

} // end namespace Engine

#endif
