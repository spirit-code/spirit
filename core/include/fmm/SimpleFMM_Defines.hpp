#pragma once
#ifndef SIMPLE_FMM_DEFINES_HPP

#include "Spirit_Defines.h"

#include <Eigen/Core>

#include <array>
#include <complex>
#include <vector>

namespace SimpleFMM
{

using Vector3c     = Eigen::Matrix<std::complex<scalar>, 3, 1>;
using Vector3      = Eigen::Matrix<scalar, 3, 1>;
using VectorXc     = Eigen::Matrix<std::complex<scalar>, Eigen::Dynamic, 1>;
using Matrix3      = Eigen::Matrix<scalar, 3, 3>;
using Matrix3c     = Eigen::Matrix<std::complex<scalar>, 3, 3>;
using MatrixXc     = Eigen::Matrix<std::complex<scalar>, Eigen::Dynamic, Eigen::Dynamic>;
using vectorfield  = std::vector<Vector3>;
using intfield     = std::vector<int>;
using scalarfield  = std::vector<scalar>;
using complexfield = std::vector<std::complex<scalar>>;

} // namespace SimpleFMM

#endif