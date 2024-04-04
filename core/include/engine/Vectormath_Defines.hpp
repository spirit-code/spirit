#pragma once

#include <Spirit/Spirit_Defines.h>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <array>
#include <complex>
#include <vector>

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
#include <thrust/universal_vector.h>

template<typename T>
using field = std::vector<T, managed_allocator<T>>;

#define SPIRIT_HOSTDEVICE __host__ __device__

template<typename T>
constexpr T * raw_pointer_cast( typename field<T>::pointer ptr ) noexcept
{
    return static_cast<T *>( ptr.get() );
}

template<typename Iter>
constexpr auto raw_pointer_cast( Iter ptr ) noexcept -> typename std::iterator_traits<Iter>::pointer
{
    return static_cast<typename std::iterator_traits<Iter>::pointer>( &( *ptr ) );
}

#else
// The general field
template<typename T>
using field = std::vector<T>;

#define SPIRIT_HOSTDEVICE

// Definition for OpenMP reduction operation using Vector3's
#pragma omp declare reduction( + : Vector3 : omp_out = omp_out + omp_in ) initializer( omp_priv = Vector3::Zero() )
#endif

// unpack thrust pointers if a raw pointer is strictly needed instead
template<typename T>
constexpr SPIRIT_HOSTDEVICE T * raw_pointer_cast( T * ptr ) noexcept
{
    return ptr;
}

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

struct PolynomialBasis
{
    Vector3 k1, k2, k3;
};

struct PolynomialTerm
{
    scalar coefficient;
    unsigned int n1, n2, n3;
};

struct AnisotropyPolynomial
{
    Vector3 k1, k2, k3;
    field<PolynomialTerm> terms;
};

struct PolynomialField
{
    field<PolynomialBasis> basis;
    field<unsigned int> site_p;
    field<PolynomialTerm> terms;
};

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
