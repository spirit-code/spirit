#pragma once
#include "Eigen/src/Core/CwiseNullaryOp.h"
#include "MatOp/SparseSymMatProd.h"
#ifndef SPIRIT_CORE_ENGINE_SOLVER_NEWTON_HPP
#define SPIRIT_CORE_ENGINE_SOLVER_NEWTON_HPP

#include "engine/Vectormath_Defines.hpp"
#include "engine/Manifoldmath.hpp"
#include <utility/Constants.hpp>

#include <SymEigsSolver.h> // Also includes <MatOp/DenseSymMatProd.h>
#include <MatOp/SparseGenMatProd.h>

#include <Eigen/IterativeLinearSolvers>

using namespace Utility;

template<>
inline void Method_Solver<Solver::Newton>::Initialize()
{
    this->forces         = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->forces_virtual = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->grad           = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->searchdir      = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );

    this->hessian_2N           = std::vector<SpMatrixX>( this->noi, SpMatrixX(2*nos, 2*nos) ); 
    this->hessian_3N_embedding = std::vector<SpMatrixX>( this->noi, SpMatrixX(3*nos, 3*nos) ); 
    this->hessian_3N_bordered  = std::vector<SpMatrixX>(this->noi, SpMatrixX(3*nos, 3*nos) );
    this->tangent_basis        = std::vector<SpMatrixX>(this->noi, SpMatrixX(3*nos, 2*nos) );

    // TODO: determine if the Hamiltonian is "square", which so far it always is, except if quadruplet interactions are present
    // This should be checked here
    this->is_square            = std::vector<bool>( this->noi, true );
    for(int img=0; img < this->noi; img++)
    {
        if(is_square[img]) // For a square Hamiltonian the embedding Hessian is independent of the spin directions, so we compute it only once
        {
            auto & image       = *this->configurations[img];
            auto & hamiltonian = this->systems[img]->hamiltonian;
            hamiltonian->Sparse_Hessian(image, hessian_3N_embedding[img]);
        }
    }

};

template<>
inline void Method_Solver<Solver::Newton>::Iteration()
{
    // update forces which are -dE/ds
    this->Calculate_Force( this->configurations, this->forces );
    this->Calculate_Force_Virtual( this->configurations, this->forces, this->forces_virtual );

    // calculate gradients for OSO
    for( int img = 0; img < this->noi; img++ )
    {

        Vectormath::set_c_a( -1, forces[img], grad[img] );

        auto & image       = *this->configurations[img];
        auto & hamiltonian = this->systems[img]->hamiltonian;

        // Compute embedding hessian
        if(!is_square[img]) // Need to recompute hessian if hamiltonian not square
        {
            // Compute embedding hessian
            hamiltonian->Sparse_Hessian(image, hessian_3N_embedding[img]);
        }

        // Compute 3N bordered Hessian
        Manifoldmath::sparse_hessian_bordered_3N(image, grad[img], hessian_3N_embedding[img], hessian_3N_bordered[img]);

        // Compute tangent basis
        Manifoldmath::sparse_tangent_basis_spherical(image, tangent_basis[img]);

        Eigen::Map<VectorX> force_vector(&(forces[img][0][0]), 3*nos, 1);
        Eigen::Map<VectorX> searchdir_vector(&(this->searchdir[img][0][0]), 3*nos, 1);

        hessian_2N[img] = (tangent_basis[img].transpose() * hessian_3N_bordered[img] * tangent_basis[img]);

        Eigen::ConjugateGradient<SpMatrixX, Eigen::Lower|Eigen::Upper> solver;
        solver.setMaxIterations(10000);
        solver.setTolerance(1e-10);
        solver.compute(hessian_2N[img]);

        // Spectrum
        /*
            Spectra::SparseGenMatProd<scalar> op( hessian_2N[img] );
            // Create and initialize a Spectra solver
            int n_modes = 2;
            Spectra::SymEigsSolver<scalar, Spectra::SMALLEST_ALGE, Spectra::SparseGenMatProd<scalar>> hessian_spectrum(&op, n_modes, 100 );
            hessian_spectrum.init();
            // Compute the specified spectrum, sorted by smallest real eigenvalue
            int nconv = hessian_spectrum.compute( 2000, 1e-10, int( Spectra::SMALLEST_ALGE ) );
            // Extract real eigenvalues
            auto eigenvalues = hessian_spectrum.eigenvalues().real();
            std::cout << "Eigenvalues " << eigenvalues.transpose() << "\n";

            MatrixX hessian2N_dense = MatrixX(hessian_2N[img]);
            std::cout << hessian2N_dense.eigenvalues() << "\n";
        */

        /* LU instead of CG
            // Eigen::SparseLU<SpMatrixX, Eigen::COLAMDOrdering<int>> solver;
            // solver.analyzePattern( hessian_2N );
            // solver.factorize( hessian_2N );
        */

        searchdir_vector = tangent_basis[img] * solver.solve(tangent_basis[img].transpose() * force_vector );

        auto conf = image.data();
        auto sd = searchdir[img].data();

        Backend::par::apply(
            nos,
            [conf, sd] SPIRIT_LAMBDA( int idx )
            {
                conf[idx] += sd[idx];
                conf[idx].normalize();
            }
        );
    }
}

template<>
inline std::string Method_Solver<Solver::Newton>::SolverName()
{
    return "Newton";
}

template<>
inline std::string Method_Solver<Solver::Newton>::SolverFullName()
{
    return "Newton";
}

#endif