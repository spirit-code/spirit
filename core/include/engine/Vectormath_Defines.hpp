#pragma once

#include <Spirit/Spirit_Defines.h>
#include <data/Pair.hpp>

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

template<typename T>
using field = std::vector<T, managed_allocator<T>>;

#define SPIRIT_HOSTDEVICE __host__ __device__

#else
// The general field
template<typename T>
using field = std::vector<T>;

#define SPIRIT_HOSTDEVICE

// Definition for OpenMP reduction operation using Vector3's
#pragma omp declare reduction( + : Vector3 : omp_out = omp_out + omp_in ) initializer( omp_priv = Vector3::Zero() )
#endif

template<typename T>
struct has_zero_method
{
private:
    template<typename U>
    static auto test(int) -> decltype(U::Zero(), std::true_type{});

    template<typename U>
    static auto test(...) -> std::false_type;
public:
    static constexpr bool value = decltype(test<T>(0))::value;
};

// utility function to get a zero valued object of type T
// this should only be useful in templated code
template<typename T>
SPIRIT_HOSTDEVICE T zero_value() noexcept
{
    if constexpr( has_zero_method<T>::value )
        return T::Zero();
    else if constexpr( std::is_arithmetic<T>::value )
        return T(0);
    else
        return T();
}

// cast an iterator to its underlying raw pointer type
template<typename Iter>
[[nodiscard]] constexpr auto raw_pointer_cast( Iter ptr ) noexcept -> typename std::iterator_traits<Iter>::pointer
{
    static_assert(
        std::is_same<typename std::decay<Iter>::type::iterator_category, std::random_access_iterator_tag>::value,
        "contiguous iterator is needed here. Otherwise there is no valid conversion" );
    return static_cast<typename std::iterator_traits<Iter>::pointer>( std::addressof( *ptr ) );
}

// noop for raw pointer types to make this operation idempotent
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
