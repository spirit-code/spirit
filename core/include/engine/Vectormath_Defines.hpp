#include <Eigen/Core>

#include <vector>

#include "Spirit_Defines.h"

// Dynamic Eigen typedefs
typedef Eigen::Matrix<scalar, -1,  1> VectorX;
typedef Eigen::Matrix<scalar,  1, -1> RowVectorX;
typedef Eigen::Matrix<scalar, -1, -1> MatrixX;

// 3D Eigen typedefs
typedef Eigen::Matrix<scalar, 3, 1> Vector3;
typedef Eigen::Matrix<scalar, 1, 3> RowVector3;
typedef Eigen::Matrix<scalar, 3, 3> Matrix3;

// Index Eigen typedefs
typedef Eigen::Matrix<int, 2, 1> indexPair;
typedef Eigen::Matrix<int, 4, 1> indexQuadruplet;

// Vectorfield and Scalarfield typedefs
#ifdef USE_CUDA
    #include "Managed_Allocator.hpp"
    typedef std::vector<int,             managed_allocator<int>>             intfield;
    typedef std::vector<scalar,          managed_allocator<scalar>>          scalarfield;
    typedef std::vector<Vector3,         managed_allocator<Vector3>>         vectorfield;
    typedef std::vector<indexPair,       managed_allocator<indexPair>>       indexPairs;
    typedef std::vector<indexQuadruplet, managed_allocator<indexQuadruplet>> indexQuadruplets;
#else
    typedef std::vector<int>             intfield;
    typedef std::vector<scalar>          scalarfield;
    typedef std::vector<Vector3>         vectorfield;
    typedef std::vector<indexPair>       indexPairs;
    typedef std::vector<indexQuadruplet> indexQuadruplets;
#endif