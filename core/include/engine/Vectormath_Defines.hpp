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
    struct Pair
    {
        int i, j, __;
        int translations[3];
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
    typedef std::vector<Pair,       managed_allocator<Pair>>       pairfield;
    typedef std::vector<Triplet,    managed_allocator<Triplet>>    tripletfield;
    typedef std::vector<Quadruplet, managed_allocator<Quadruplet>> quadrupletfield;
#else
    typedef std::vector<int>             intfield;
    typedef std::vector<scalar>          scalarfield;
    typedef std::vector<Vector3>         vectorfield;
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
    struct Neighbour
    {
        // Basis indices of atom and neighbour
        int iatom, ineigh;
        // Shell index
        int idx_shell;
        // Translations of the neighbour cell
        std::array<int,3> translations;
    };
    typedef std::vector<Pair>       pairfield;
    typedef std::vector<Triplet>    tripletfield;
    typedef std::vector<Quadruplet> quadrupletfield;
    typedef std::vector<Neighbour>  neighbourfield;
#endif