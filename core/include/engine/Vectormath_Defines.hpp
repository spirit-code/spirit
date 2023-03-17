#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <array>
#include <complex>
#include <vector>

#include "Spirit_Defines.h"

// Dynamic Eigen typedefs
using VectorX    = Eigen::Matrix<scalar, -1, 1>;
using RowVectorX = Eigen::Matrix<scalar, 1, -1>;
using MatrixX    = Eigen::Matrix<scalar, -1, -1>;
using SpMatrixX  = Eigen::SparseMatrix<scalar>;

// 3D Eigen typedefs
using Vector3    = Eigen::Matrix<scalar, 3, 1>;
using RowVector3 = Eigen::Matrix<scalar, 1, 3>;
using Matrix3    = Eigen::Matrix<scalar, 3, 3>;

using Vector3c = Eigen::Matrix<std::complex<scalar>, 3, 1>;
using Matrix3c = Eigen::Matrix<std::complex<scalar>, 3, 3>;

// 2D Eigen typedefs
using Vector2 = Eigen::Matrix<scalar, 2, 1>;

// Different definitions for regular C++ and CUDA
#ifdef SPIRIT_USE_CUDA
// The general field, using the managed allocator
#include "Managed_Allocator.hpp"
template<typename T>
using field = std::vector<T, managed_allocator<T>>;

struct Site
{
    // Basis index
    int i;
    // Translations of the basis cell
    int translations[3];
};
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
#else
// The general field
template<typename T>
using field = std::vector<T>;

struct Site
{
    // Basis index
    int i;
    // Translations of the basis cell
    std::array<int, 3> translations;
};
struct Pair
{
    int i, j;
    std::array<int, 3> translations;
};
struct Triplet
{
    int i, j, k;
    std::array<int, 3> d_j, d_k;
};
struct Quadruplet
{
    int i, j, k, l;
    std::array<int, 3> d_j, d_k, d_l;
};

// Definition for OpenMP reduction operation using Vector3's
#pragma omp declare reduction( + : Vector3 : omp_out = omp_out + omp_in ) initializer( omp_priv = Vector3::Zero() )
#endif

struct Neighbour : Pair
{
    // Shell index
    int idx_shell;
};

// Important fields
using intfield    = field<int>;
using scalarfield = field<scalar>;
using vectorfield = field<Vector3>;

// Additional fields
using pairfield       = field<Pair>;
using tripletfield    = field<Triplet>;
using quadrupletfield = field<Quadruplet>;
using neighbourfield  = field<Neighbour>;
using vector2field    = field<Vector2>;