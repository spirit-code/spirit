/// EIGEN
#include <Eigen/Core>

#include "Core_Defines.h"

typedef Eigen::Matrix<scalar, -1, 1> VectorX;
typedef Eigen::Matrix<scalar, 1, -1> RowVectorX;

typedef Eigen::Matrix<scalar, 3, 1> Vector3;
typedef Eigen::Matrix<scalar, 1, 3> RowVector3;

typedef Eigen::Matrix<scalar, -1, -1> MatrixX;

typedef Eigen::Matrix<scalar, 3, 3> Matrix3;

typedef Eigen::Matrix<scalar, -1, 3> VectorField; // (N, 3)

// template <size_t N>
// using Vector = Matrix<N, 1>;