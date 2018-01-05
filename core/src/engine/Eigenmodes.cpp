#include <engine/Eigenmodes.hpp>
#include <engine/Manifoldmath.hpp>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <GenEigsSolver.h>  // Also includes <MatOp/DenseGenMatProd.h>
#include <GenEigsRealShiftSolver.h>

namespace Engine
{
    namespace Eigenmodes
    {
        bool Hessian_Full_Spectrum(const vectorfield & spins, const vectorfield & gradient, const MatrixX & hessian,
            MatrixX & tangent_basis, MatrixX & hessian_constrained, VectorX & eigenvalues, MatrixX & eigenvectors)
        {
            int nos = spins.size();

            // Calculate the final Hessian to use for the minimum mode
            // TODO: add option to choose different Hessian calculation
            hessian_constrained = MatrixX::Zero(2*nos, 2*nos);
            tangent_basis       = MatrixX::Zero(3*nos, 2*nos);
            Manifoldmath::hessian_bordered(spins, gradient, hessian, tangent_basis, hessian_constrained);
            // Manifoldmath::hessian_projected(spins, grad, hess, tangent_basis, hessian_constrained);
            // Manifoldmath::hessian_weingarten(spins, grad, hess, tangent_basis, hessian_constrained);
            // Manifoldmath::hessian_spherical(spins, grad, hess, tangent_basis, hessian_constrained);
            // Manifoldmath::hessian_covariant(spins, grad, hess, tangent_basis, hessian_constrained);
            
            // Create and initialize a Eigen solver
            Eigen::SelfAdjointEigenSolver<MatrixX> hessian_spectrum(hessian_constrained);

            // Extract real eigenvalues
            eigenvalues = hessian_spectrum.eigenvalues().real();
            // Retrieve the real eigenvectors
            eigenvalues = hessian_spectrum.eigenvectors().real();

            // Return whether the calculation was successful
            return true;
        }

        bool Hessian_Partial_Spectrum(const vectorfield & spins, const vectorfield & gradient, const MatrixX & hessian, int n_modes,
            MatrixX & tangent_basis, MatrixX & hessian_constrained, VectorX & eigenvalues, MatrixX & eigenvectors)
        {
            int nos = spins.size();
            n_modes = std::max(0, std::min(2*nos-1, n_modes));

            // Calculate the final Hessian to use for the minimum mode
            // TODO: add option to choose different Hessian calculation
            hessian_constrained = MatrixX::Zero(2*nos, 2*nos);
            tangent_basis       = MatrixX::Zero(3*nos, 2*nos);
            Manifoldmath::hessian_bordered(spins, gradient, hessian, tangent_basis, hessian_constrained);
            // Manifoldmath::hessian_projected(spins, grad, hess, tangent_basis, hessian_constrained);
            // Manifoldmath::hessian_weingarten(spins, grad, hess, tangent_basis, hessian_constrained);
            // Manifoldmath::hessian_spherical(spins, grad, hess, tangent_basis, hessian_constrained);
            // Manifoldmath::hessian_covariant(spins, grad, hess, tangent_basis, hessian_constrained);
            
            // Create the Spectra Matrix product operation
            Spectra::DenseGenMatProd<scalar> op(hessian_constrained);
            // Create and initialize a Spectra solver
            Spectra::GenEigsSolver< scalar, Spectra::SMALLEST_REAL, Spectra::DenseGenMatProd<scalar> > hessian_spectrum(&op, n_modes, 2*nos);
            hessian_spectrum.init();

            // Compute the specified spectrum, sorted by smallest real eigenvalue
            int nconv = hessian_spectrum.compute(1000, 1e-10, int(Spectra::SMALLEST_REAL));

            // Extract real eigenvalues
            eigenvalues = hessian_spectrum.eigenvalues().real();

            // Retrieve the real eigenvectors
            eigenvectors = hessian_spectrum.eigenvectors().real();

            // Return whether the calculation was successful
            return (hessian_spectrum.info() == Spectra::SUCCESSFUL) && (nconv > 0);
        }
    }
}