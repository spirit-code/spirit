#pragma once
#ifndef EIGENMODES_H
#define EIGENMODES_H

#include "Spirit_Defines.h"
#include <data/Geometry.hpp>
#include <data/Parameters_Method.hpp>
#include <engine/Vectormath_Defines.hpp>

#include <vector>
#include <memory>

namespace Engine
{
    namespace Eigenmodes
    {
        // Calculate the full eigenspectrum of a Hessian (needs to be self-adjoint)
        // gradient and hessian should be the 3N-dimensional representations without constraints
        bool Hessian_Full_Spectrum(const std::shared_ptr<Data::Parameters_Method> parameters,
            const vectorfield & spins, const vectorfield & gradient, const MatrixX & hessian,
            MatrixX & tangent_basis, MatrixX & hessian_constrained, VectorX & eigenvalues, MatrixX & eigenvectors);

        // Calculate a partial eigenspectrum of a Hessian
        // gradient and hessian should be the 3N-dimensional representations without constraints
        bool Hessian_Partial_Spectrum(const std::shared_ptr<Data::Parameters_Method> parameters,
            const vectorfield & spins, const vectorfield & gradient, const MatrixX & hessian, int n_modes,
            MatrixX & tangent_basis, MatrixX & hessian_constrained, VectorX & eigenvalues, MatrixX & eigenvectors);
    };// end namespace Eigenmodes
}// end namespace Engine
#endif