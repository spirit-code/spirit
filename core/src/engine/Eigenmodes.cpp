#include <engine/Eigenmodes.hpp>
#include <engine/Manifoldmath.hpp>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <SymEigsSolver.h>  // Also includes <MatOp/DenseSymMatProd.h>

namespace Engine
{
    namespace Eigenmodes
    {
        bool Hessian_Full_Spectrum(const std::shared_ptr<Data::Parameters_Method> parameters,
            const vectorfield & spins, const vectorfield & gradient, const MatrixX & hessian,
            MatrixX & tangent_basis, MatrixX & hessian_constrained, VectorX & eigenvalues, MatrixX & eigenvectors)
        {
            int nos = spins.size();

            // Calculate the final Hessian to use for the minimum mode
            // TODO: add option to choose different Hessian calculation
            hessian_constrained = MatrixX::Zero(2*nos, 2*nos);
            tangent_basis       = MatrixX::Zero(3*nos, 2*nos);
            Manifoldmath::hessian_bordered(spins, gradient, hessian, tangent_basis, hessian_constrained);
            // Manifoldmath::hessian_projected(spins, gradient, hessian, tangent_basis, hessian_constrained);
            // Manifoldmath::hessian_weingarten(spins, gradient, hessian, tangent_basis, hessian_constrained);
            // Manifoldmath::hessian_spherical(spins, gradient, hessian, tangent_basis, hessian_constrained);
            // Manifoldmath::hessian_covariant(spins, gradient, hessian, tangent_basis, hessian_constrained);
            
            // Create and initialize a Eigen solver. Note: the hessian matrix should be symmetric!
            Eigen::SelfAdjointEigenSolver<MatrixX> hessian_spectrum(hessian_constrained);

            // Extract real eigenvalues
            eigenvalues = hessian_spectrum.eigenvalues().real();
            // Retrieve the real eigenvectors
            eigenvectors = hessian_spectrum.eigenvectors().real();

            // Return whether the calculation was successful
            return true;
        }

        bool Hessian_Partial_Spectrum(const std::shared_ptr<Data::Parameters_Method> parameters,
            const vectorfield & spins, const vectorfield & gradient, const MatrixX & hessian, int n_modes,
            MatrixX & tangent_basis, MatrixX & hessian_constrained, VectorX & eigenvalues, MatrixX & eigenvectors)
        {
            int nos = spins.size();

            // Restrict number of calculated modes to [1,2N)
            n_modes = std::max(1, std::min(2*nos-2, n_modes));

            // If we have only one spin, we can only calculate the full spectrum
            if (n_modes == nos)
                return Hessian_Full_Spectrum(parameters, spins, gradient, hessian, tangent_basis, hessian_constrained, eigenvalues, eigenvectors);

            // Calculate the final Hessian to use for the minimum mode
            // TODO: add option to choose different Hessian calculation
            hessian_constrained = MatrixX::Zero(2*nos, 2*nos);
            tangent_basis       = MatrixX::Zero(3*nos, 2*nos);
            Manifoldmath::hessian_bordered(spins, gradient, hessian, tangent_basis, hessian_constrained);
            // Manifoldmath::hessian_projected(spins, gradient, hessian, tangent_basis, hessian_constrained);
            // Manifoldmath::hessian_weingarten(spins, gradient, hessian, tangent_basis, hessian_constrained);
            // Manifoldmath::hessian_spherical(spins, gradient, hessian, tangent_basis, hessian_constrained);
            // Manifoldmath::hessian_covariant(spins, gradient, hessian, tangent_basis, hessian_constrained);
            
            // Remove degrees of freedom of pinned spins
            #ifdef SPIRIT_ENABLE_PINNING
                for (int i=0; i<nos; ++i)
                {
                    if (!parameters->pinning->mask_unpinned[i])
                    {
                        // Remove interaction block
                        for (int j=0; j<nos; ++j)
                        {
                            hessian_constrained.block<2,2>(2*i,2*j).setZero();
                            hessian_constrained.block<2,2>(2*j,2*i).setZero();
                        }
                        // Set diagonal matrix entries of pinned spins to a large value
                        hessian_constrained.block<2,2>(2*i,2*i).setZero();
                        hessian_constrained.block<2,2>(2*i,2*i).diagonal().setConstant(nos*1e5);
                    }
                }
            #endif // SPIRIT_ENABLE_PINNING

            // Create the Spectra Matrix product operation
            Spectra::DenseSymMatProd<scalar> op(hessian_constrained);
            // Create and initialize a Spectra solver
            Spectra::SymEigsSolver< scalar, Spectra::SMALLEST_ALGE, Spectra::DenseSymMatProd<scalar> > hessian_spectrum(&op, n_modes, 2*nos);
            hessian_spectrum.init();

            // Compute the specified spectrum, sorted by smallest real eigenvalue
            int nconv = hessian_spectrum.compute(1000, 1e-10, int(Spectra::SMALLEST_ALGE));

            // Extract real eigenvalues
            eigenvalues = hessian_spectrum.eigenvalues().real();

            // Retrieve the real eigenvectors
            eigenvectors = hessian_spectrum.eigenvectors().real();

            // Return whether the calculation was successful
            return (hessian_spectrum.info() == Spectra::SUCCESSFUL) && (nconv > 0);
        }
    }
}
