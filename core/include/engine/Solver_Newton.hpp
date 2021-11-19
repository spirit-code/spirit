#pragma once
#include "Eigen/src/Core/CwiseNullaryOp.h"
#include "MatOp/SparseSymMatProd.h"
#ifndef SPIRIT_CORE_ENGINE_SOLVER_NEWTON_HPP
#define SPIRIT_CORE_ENGINE_SOLVER_NEWTON_HPP

#include "engine/Vectormath_Defines.hpp"
#include "engine/Manifoldmath.hpp"
#include "engine/Solver_Kernels.hpp"

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
    this->searchdir_2N   = std::vector<VectorX>( this->noi, VectorX::Zero(2*nos) );

    this->hessian_2N           = std::vector<SpMatrixX>( this->noi, SpMatrixX(2*nos, 2*nos) ); 
    this->hessian_3N_embedding = std::vector<SpMatrixX>( this->noi, SpMatrixX(3*nos, 3*nos) ); 
    this->hessian_3N_bordered  = std::vector<SpMatrixX>(this->noi, SpMatrixX(3*nos, 3*nos) );
    this->tangent_basis        = std::vector<SpMatrixX>(this->noi, SpMatrixX(3*nos, 2*nos) );

    this->temp1 = vectorfield(this->nos, {0,0,0}); // temporary field for linesearch

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

        hessian_2N[img] = (tangent_basis[img].transpose() * hessian_3N_bordered[img] * tangent_basis[img]).eval();

        // /* LU instead of CG (way to slow)
        // hessian_2N[img].makeCompressed();
        // Eigen::SparseLU<SpMatrixX, Eigen::COLAMDOrdering<int>> solver;
        // solver.analyzePattern( hessian_2N[img] );
        // solver.factorize( hessian_2N[img] );
        // this->searchdir_2N[img] = solver.solve(force_2N);

        hessian_2N[img].makeCompressed();

        fmt::print("Solving inverse Hessian\n");

        Eigen::ConjugateGradient<SpMatrixX, Eigen::Lower|Eigen::Upper> solver;

        solver.compute(hessian_2N[img]);
        solver.setTolerance(1e-4);

        VectorX force_2N = (tangent_basis[img].transpose() * force_vector).eval();
        this->searchdir_2N[img] = solver.solveWithGuess(force_2N, this->searchdir_2N[img]);
        std::cout << "#iterations:     " << solver.iterations() << std::endl;
        std::cout << "estimated error: " << solver.error()      << std::endl;

        searchdir_vector = tangent_basis[img] * searchdir_2N[img];
        scalar linear_coeff_delta_e = searchdir_2N[img].dot(-force_2N);
        scalar quadratic_coeff_delta_e = -linear_coeff_delta_e/2; // for newtons method this is true

        fmt::print("Finished solving inverse Hessian\n");

        scalar alpha = Solver_Kernels::backtracking_linesearch(
            *hamiltonian,
            searchdir[img],
            linear_coeff_delta_e,
            quadratic_coeff_delta_e,
            0.15,
            0.5,
            image,
            temp1,
            energy_buffer_current,
            energy_buffer_step
        );

        fmt::print("alpha = {:.6f}\n", alpha);

        auto conf = image.data();
        auto sd = searchdir[img].data();
        Backend::par::apply(
            nos,
            [conf, sd, alpha] SPIRIT_LAMBDA( int idx )
            {
                conf[idx] += alpha * sd[idx];
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