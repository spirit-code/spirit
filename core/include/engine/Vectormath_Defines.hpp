#include <Eigen/Core>

#include <vector>

#include "Core_Defines.h"

// Dynamic Eigen typedefs
typedef Eigen::Matrix<scalar, -1, 1> VectorX;
typedef Eigen::Matrix<scalar, 1, -1> RowVectorX;
typedef Eigen::Matrix<scalar, -1, -1> MatrixX;

// 3D Eigen typedefs
typedef Eigen::Matrix<scalar, 3, 1> Vector3;
typedef Eigen::Matrix<scalar, 1, 3> RowVector3;
typedef Eigen::Matrix<scalar, 3, 3> Matrix3;

// Vectorfield and Scalarfield typedefs
#ifdef USE_CUDA
    #include "managed_allocator.hpp"
    typedef std::vector<Vector3, managed_allocator<Vector3>> vectorfield;
    typedef std::vector<scalar, managed_allocator<scalar>> scalarfield;
#else
    typedef std::vector<Vector3> vectorfield;
    typedef std::vector<scalar> scalarfield;
#endif