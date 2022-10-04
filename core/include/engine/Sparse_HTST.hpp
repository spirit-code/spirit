#pragma once
#ifndef SPIRIT_CORE_ENGINE_SPARSE_HTST_HPP
#define SPIRIT_CORE_ENGINE_SPARSE_HTST_HPP
#ifndef SPIRIT_SKIP_HTST

#include "Spirit_Defines.h"
#include <data/Spin_System.hpp>
#include <data/Spin_System_Chain.hpp>
#include <engine/Vectormath_Defines.hpp>

#include <fstream>
#include <iostream>

namespace Engine
{
namespace Sparse_HTST
{

// A small function for debugging purposes
inline void saveMatrix( std::string fname, const SpMatrixX & matrix )
{
    std::cout << "Saving matrix to file: " << fname << "\n";
    std::ofstream file( fname );
    if( file && file.is_open() )
    {
        file << matrix;
    }
    else
    {
        std::cerr << "Could not save matrix!";
    }
}

template<typename Field_Like>
inline void saveField( std::string fname, const Field_Like & field )
{
    int n = field.size();
    std::cout << "Saving field to file: " << fname << "\n";
    std::ofstream file( fname );
    if( file && file.is_open() )
    {
        for( int i = 0; i < n; i++ )
            file << field[i] << "\n";
    }
    else
    {
        std::cerr << "Could not save field!";
    }
}

// Note the two images should correspond to one minimum and one saddle point
void Calculate( Data::HTST_Info & htst_info );

// Calculate the sparse Velocity matrix
void Sparse_Calculate_Dynamical_Matrix(
    const vectorfield & spins, const scalarfield & mu_s, const SpMatrixX & hessian, SpMatrixX & velocity );

void Sparse_Geodesic_Eigen_Decomposition(
    const vectorfield & image, const vectorfield & gradient, const SpMatrixX & hessian, SpMatrixX & hessian_geodesic_3N,
    SpMatrixX & hessian_geodesic_2N, SpMatrixX & tangent_basis, VectorX & eigenvalues, MatrixX & eigenvectors );

void sparse_hessian_bordered_3N(
    const vectorfield & image, const vectorfield & gradient, const SpMatrixX & hessian, SpMatrixX & hessian_out );

} // namespace Sparse_HTST
} // namespace Engine

#endif
#endif