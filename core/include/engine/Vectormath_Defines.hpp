#pragma once

#include <Eigen/Core>

#include <vector>
#include <array>

#include "Spirit_Defines.h"

// Dynamic Eigen typedefs
typedef Eigen::Matrix<scalar, -1,  1> VectorX;
typedef Eigen::Matrix<scalar,  1, -1> RowVectorX;
typedef Eigen::Matrix<scalar, -1, -1> MatrixX;

// 3D Eigen typedefs
typedef Eigen::Matrix<scalar, 3, 1> Vector3;
typedef Eigen::Matrix<scalar, 1, 3> RowVector3;
typedef Eigen::Matrix<scalar, 3, 3> Matrix3;

// Vectorfield and Scalarfield typedefs
#ifdef USE_CUDA
    #include "Managed_Allocator.hpp"
    typedef std::vector<int,             managed_allocator<int>>             intfield;
    typedef std::vector<scalar,          managed_allocator<scalar>>          scalarfield;
    typedef std::vector<Vector3,         managed_allocator<Vector3>>         vectorfield;
    typedef std::vector<Matrix3,         managed_allocator<Matrix3>>         matrixfield;
    struct Pair
    {
        // Basis indices of first and second atom of pair
        int i, j;
        // Translations of the basis cell of second atom of pair
        int translations[3];
    };
    struct Triplet
    {
        int i, j, k;
        int d_j[3], d_k[3];
    };
    struct Quadruplet
    {
        int i, j, k, l;
        int d_j[3], d_k[3], d_l[3];
    };
    struct Neighbour : Pair
    {
        // Shell index
        int idx_shell;
    };
    typedef std::vector<Pair,       managed_allocator<Pair>>       pairfield;
    typedef std::vector<Triplet,    managed_allocator<Triplet>>    tripletfield;
    typedef std::vector<Quadruplet, managed_allocator<Quadruplet>> quadrupletfield;
    typedef std::vector<Neighbour,  managed_allocator<Neighbour>>  neighbourfield;
#else
    typedef std::vector<int>     intfield;
    typedef std::vector<scalar>  scalarfield;
    typedef std::vector<Vector3> vectorfield;
    typedef std::vector<Matrix3> matrixfield;
    struct Pair
    {
        int i, j;
        std::array<int,3> translations;
    };
    struct Triplet
    {
        int i, j, k;
        std::array<int,3> d_j, d_k;
    };
    struct Quadruplet
    {
        int i, j, k, l;
        std::array<int,3> d_j, d_k, d_l;
    };
    struct Neighbour : Pair
    {
        // Shell index
        int idx_shell;
    };
    typedef std::vector<Pair>       pairfield;
    typedef std::vector<Triplet>    tripletfield;
    typedef std::vector<Quadruplet> quadrupletfield;
    typedef std::vector<Neighbour>  neighbourfield;

    // Definition for OpenMP reduction operation using Vector3's
    #pragma omp declare reduction (+: Vector3: omp_out=omp_out+omp_in)\
        initializer(omp_priv=Vector3::Zero())
#endif