#include <engine/HTST.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Manifoldmath.hpp>
#include <engine/Hamiltonian_Heisenberg.hpp>
#include <utility/Constants.hpp>
#include <utility/Logging.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
// #include <Eigen/Array>
#include <Eigen/Eigenvalues>
//#include <unsupported/Eigen/CXX11/Tensor>
#include <GenEigsSolver.h>  // Also includes <MatOp/DenseGenMatProd.h>
#include <GenEigsRealShiftSolver.h>

#include <fmt/format.h>

#include <iostream>

namespace C = Utility::Constants;

namespace Engine
{
    namespace HTST
    {
        // Note the two images should correspond to one minimum and one saddle point
        // Non-extremal images may yield incorrect Hessians and thus incorrect results
        void Calculate_Prefactor(Data::HTST_Info & htst_info)
        {
            std::cerr << "Getting Prefactor" << std::endl;

            bool is_afm = false;

            scalar epsilon = 1e-4;

            ////////////////////////////////////////////////////////////////////////
            // Initial state minimum
            std::cerr << std::endl << "Calculation for the Minimum" << std::endl;

            // The gradient (unprojected)
            std::cerr << "    Evaluation of the gradient..." << std::endl;
            auto& image_minimum = *htst_info.minimum->spins;
            int nos = image_minimum.size();
            vectorfield gradient_minimum(nos, {0,0,0});
            htst_info.minimum->hamiltonian->Gradient(image_minimum, gradient_minimum);

            // Evaluation of the Hessian...
            MatrixX hessian_minimum = MatrixX::Zero(3*nos,3*nos);
            std::cerr << "    Evaluation of the Hessian..." << std::endl;
            htst_info.minimum->hamiltonian->Hessian(image_minimum, hessian_minimum);

            // Eigendecomposition
            std::cerr << "    Eigendecomposition..." << std::endl;
            MatrixX hessian_geodesic_minimum_3N(3*nos, 3*nos);
            MatrixX hessian_geodesic_minimum_2N(2*nos, 2*nos);
            VectorX eigenvalues_minimum = VectorX::Zero(2*nos);
            MatrixX eigenvectors_minimum = MatrixX::Zero(2*nos, 2*nos);
            Geodesic_Eigen_Decomposition(image_minimum, gradient_minimum, hessian_minimum,
                hessian_geodesic_minimum_3N, hessian_geodesic_minimum_2N, eigenvalues_minimum, eigenvectors_minimum);

            // Print some eigenvalues
            std::cerr << "10 lowest eigenvalues at minimum:" << std::endl;
            for (int i=0; i<10; ++i)
                std::cerr << fmt::format("ew[{}]={:^20e}   ew[{}]={:^20e}", i, eigenvalues_minimum[i], i+2*nos-10, eigenvalues_minimum[i+2*nos-10]) << std::endl;

            // Check for eigenvalues < 0 (i.e. not a minimum)
            std::cerr << "Checking if actually a minimum..." << std::endl;
            bool minimum = true;
            for (int i=0; i < eigenvalues_minimum.size(); ++i)
                if (eigenvalues_minimum[i] < -epsilon) minimum = false;
            if (!minimum) std::cerr << "WARNING: NOT A MINIMUM!!" << std::endl;

            // Checking for zero modes at the minimum...
            std::cerr << "Checking for zero modes at the minimum..." << std::endl;
            int n_zero_modes_minimum = 0;
            for (int i=0; i < eigenvalues_minimum.size(); ++i)
                if (std::abs(eigenvalues_minimum[i]) <= epsilon) ++n_zero_modes_minimum;

            // Deal with zero modes if any (calculate volume)
            scalar volume_minimum = 1;
            if (n_zero_modes_minimum > 0)
            {
                std::cerr << fmt::format("ZERO MODES AT MINIMUM (N={})", n_zero_modes_minimum) << std::endl;

                if (is_afm)
                    volume_minimum = Calculate_Zero_Volume(htst_info.minimum);
                else
                    volume_minimum = Calculate_Zero_Volume(htst_info.minimum);
            }

            ////////////////////////////////////////////////////////////////////////
            // Saddle point
            std::cerr << std::endl << "Calculation for the Saddle Point" << std::endl;

            // The gradient (unprojected)
            std::cerr << "    Evaluation of the gradient..." << std::endl;
            auto& image_sp = *htst_info.saddle_point->spins;
            vectorfield gradient_sp(nos, {0,0,0});
            htst_info.saddle_point->hamiltonian->Gradient(image_sp, gradient_sp);

            // Evaluation of the Hessian...
            MatrixX hessian_sp = MatrixX::Zero(3*nos,3*nos);
            std::cerr << "    Evaluation of the Hessian..." << std::endl;
            htst_info.saddle_point->hamiltonian->Hessian(image_sp, hessian_sp);

            // Eigendecomposition
            std::cerr << "    Eigendecomposition..." << std::endl;
            MatrixX hessian_geodesic_sp_3N(3*nos, 3*nos);
            MatrixX hessian_geodesic_sp_2N(2*nos, 2*nos);
            VectorX eigenvalues_sp = VectorX::Zero(2*nos);
            MatrixX eigenvectors_sp = MatrixX::Zero(2*nos, 2*nos);
            Geodesic_Eigen_Decomposition(image_sp, gradient_sp, hessian_sp,
                hessian_geodesic_sp_3N, hessian_geodesic_sp_2N, eigenvalues_sp, eigenvectors_sp);

            // Print some eigenvalues
            std::cerr << "10 lowest eigenvalues at saddle point:" << std::endl;
            for (int i=0; i<10; ++i)
                std::cerr << fmt::format("ew[{}]={:^20e}   ew[{}]={:^20e}", i, eigenvalues_sp[i], i+2*nos-10, eigenvalues_sp[i+2*nos-10]) << std::endl;

            // Check if lowest eigenvalue < 0 (else it's not a SP)
            std::cerr << "Checking if actually a saddle point..." << std::endl;
            bool saddlepoint = false;
            if (eigenvalues_sp[0] < -epsilon) saddlepoint = true;  // QUESTION: is ev[0] the lowest? should I use Spectra here?
            if (!saddlepoint) std::cerr << "NOT A SADDLE POINT" << std::endl;

            // Check if second-lowest eigenvalue < 0 (higher-order SP)
            if (saddlepoint)
            {
                std::cerr << "Checking if higher order saddle point..." << std::endl;
                int n_negative = 0;
                for (int i=0; i < eigenvalues_sp.size(); ++i)
                    if (eigenvalues_sp[i] < -epsilon) ++n_negative;

                if (n_negative > 1)
                    std::cerr << fmt::format("WARNING: HIGHER ORDER SADDLE POINT (N={})", n_negative) << std::endl;
            }

            // Checking for zero modes at the saddle point...
            std::cerr << "Checking for zero modes at the saddle point..." << std::endl;
            int n_zero_modes_sp = 0;
            for (int i=0; i < eigenvalues_sp.size(); ++i)
                if (std::abs(eigenvalues_sp[i]) <= epsilon) ++n_zero_modes_sp;

            // Deal with zero modes if any (calculate volume)
            scalar volume_sp = 1;
            if (n_zero_modes_sp > 0)
            {
                std::cerr << fmt::format("ZERO MODES AT SADDLE POINT (N={})", n_zero_modes_sp) << std::endl;

                if (is_afm)
                    volume_sp = Calculate_Zero_Volume(htst_info.saddle_point);
                else
                    volume_sp = Calculate_Zero_Volume(htst_info.saddle_point);
            }

            ////////////////////////////////////////////////////////////////////////
            std::cerr << "Calculating perpendicular velocity ('a' factors)..." << std::endl;
            // Calculation of the 'a' parameters...
            VectorX perpendicular_velocity_sp(2*nos);
            MatrixX basis_sp(3*nos, 2*nos);
            Manifoldmath::tangent_basis_spherical(image_sp, basis_sp);
            // Manifoldmath::tangent_basis(image_sp, basis_sp);
            // TODO
            // Calculate_Perpendicular_Velocity_2N(image_sp, hessian_geodesic_sp_2N, basis_sp, eigenvectors_sp, perpendicular_velocity_sp);
            Calculate_Perpendicular_Velocity(image_sp, htst_info.saddle_point->geometry->mu_s, hessian_geodesic_sp_3N, basis_sp, eigenvectors_sp, perpendicular_velocity_sp);
            // QUESTION: is scaling perpendicular_velocity_sp with mub/mry necessary?

            ////////////////////////////////////////////////////////////////////////
            // Calculation of the prefactor...
            std::cerr << "Calculating prefactor..." << std::endl;
            scalar prefactor, exponent;
            // Calculate_Prefactor(nos, n_zero_modes_minimum, n_zero_modes_sp, volume_minimum, volume_sp, eigenvalues_minimum, eigenvalues_sp, perpendicular_velocity_sp,
            //     prefactor, exponent);

            // Calculate the exponent for the temperature-dependence of the prefactor
            //      The exponent depends on the number of zero modes at the different states
            exponent = 0.5 * (n_zero_modes_minimum - n_zero_modes_sp);

            // QUESTION: g_e is the electron's g-factor [unitless] -> mu_s = g_e * mu_B / hbar * Spin
            scalar g_e = 2.00231930436182;

            // Calculate "me" - QUESTION: what is this?
            scalar me = 1;
            for (int i=0; i < (n_zero_modes_minimum - n_zero_modes_sp); ++i)
                me *= 2*C::Pi * C::k_B;
            me = std::sqrt(me);

            // Calculate Omega_0, i.e. the entropy contribution
            // TODO: the `n_zero_modes_sp+1` should be `n_zero_modes_sp+n_negative_modes_sp`
            scalar Omega_0 = 1;
            if( n_zero_modes_minimum > n_zero_modes_sp+1 )
            {
                for( int i = n_zero_modes_sp+1; i<n_zero_modes_minimum; ++i )
                    Omega_0 /= std::sqrt(eigenvalues_sp[i]);
            }
            else if( n_zero_modes_minimum < n_zero_modes_sp+1 )
            {
                for( int i = n_zero_modes_minimum; i<(n_zero_modes_sp+1); ++i )
                    Omega_0 *= std::sqrt(eigenvalues_minimum[i]);
            }
            for( int i=std::max(n_zero_modes_minimum, n_zero_modes_sp+1); i < 2*nos; ++i )
                Omega_0 *= std::sqrt(eigenvalues_minimum[i] / eigenvalues_sp[i]);
            // Omega_0 = std::sqrt(Omega_0);

            // Calculate "s" - QUESTION: what is it?
            scalar s = 0;
            for (int i = n_zero_modes_sp+1; i < 2*nos; ++i)
                s += std::pow(perpendicular_velocity_sp[i], 2) / eigenvalues_sp[i];
            s = std::sqrt(s);

            // Calculate the prefactor
            scalar v = me * volume_sp / volume_minimum * s;
            prefactor = g_e / (C::hbar * 1e-12) * Omega_0 * v / ( 2*C::Pi );

            std::cerr << fmt::format("exponent    = {:^20e}", exponent)<< std::endl;
            std::cerr << fmt::format("me          = {:^20e}", me) << std::endl;
            std::cerr << fmt::format("m = Omega_0 = {:^20e}", Omega_0) << std::endl;
            std::cerr << fmt::format("s           = {:^20e}", s) << std::endl;
            std::cerr << fmt::format("volume_sp   = {:^20e}", volume_sp) << std::endl;
            std::cerr << fmt::format("volume_min  = {:^20e}", volume_minimum) << std::endl;
            std::cerr << fmt::format("hbar[meV*s] = {:^20e}", C::hbar*1e-12) << std::endl;
            std::cerr << fmt::format("v = dynamical prefactor = {:^20e}", v) << std::endl;
            std::cerr << fmt::format("prefactor               = {:^20e}", prefactor) << std::endl;

            ////////////////////////////////////////////////////////////////////////
            htst_info.eigenvalues_min.assign(eigenvalues_minimum.data(), eigenvalues_minimum.data() + 2*nos);
            // htst_info.eigenvectors_min.assign(eigenvectors_minimum.data(), eigenvectors_minimum.data() + 2*3*nos);
            htst_info.eigenvalues_sp.assign(eigenvalues_sp.data(), eigenvalues_sp.data() + 2*nos);
            // htst_info.eigenvectors_sp.assign(eigenvectors_sp.data(), eigenvectors_sp.data() + 2*3*nos);
            //     htst_info.eigenvectors_min = std::vector<Vector3>(2*nos);
            // for( int i=0; i<2*nos; ++i )
            // {
            //     htst_info.eigenvectors_min[i] = eigenvectors_minimum[i];
            // }

            htst_info.temperature_exponent = exponent;
            htst_info.volume_min           = volume_minimum;
            htst_info.volume_sp            = volume_sp;
            htst_info.me                   = me;
            htst_info.s                    = s;
            htst_info.Omega_0              = Omega_0;
            htst_info.prefactor_dynamical  = v;
            htst_info.prefactor            = prefactor;
        }

        // TODO: this does not work in the 2d hexagonal case...
        scalar Calculate_Zero_Volume(const std::shared_ptr<Data::Spin_System> system)
        {
            int   nos             = system->geometry->nos;
            auto& n_cells         = system->geometry->n_cells;
            auto& spins           = *system->spins;
            auto& spin_positions  = system->geometry->positions;
            auto& geometry        = *system->geometry;
            auto& bravais_vectors = system->geometry->bravais_vectors;

            // Dimensionality of the zero mode
            int zero_mode_dimensionality = 0;
            Vector3 zero_mode_length{0,0,0};
            for( int ibasis=0; ibasis<3; ++ibasis )
            {
                // Only a periodical direction can be a true zero mode
                if( system->hamiltonian->boundary_conditions[ibasis] && geometry.n_cells[ibasis] > 1 )
                {
                    // Vector3 shift_pos, test_pos;
                    vectorfield spins_shifted(nos, Vector3{0,0,0});

                    int shift = 0;
                    if( ibasis == 0 )
                        shift = geometry.n_cell_atoms;
                    else if( ibasis == 1 )
                        shift = geometry.n_cell_atoms * geometry.n_cells[0];
                    else if( ibasis == 2 )
                        shift = geometry.n_cell_atoms * geometry.n_cells[0] * geometry.n_cells[1];

                    for (int isite = 0; isite < nos; ++isite)
                    {
                        spins_shifted[(isite + shift) % nos] = spins[isite];
                    }

                    zero_mode_length[ibasis] = Manifoldmath::dist_geodesic(spins, spins_shifted);

                    // Increment zero mode dimensionality
                    ++zero_mode_dimensionality;
                }
            }

            // Calculate the volume depending on the number of periodical boundaries
            scalar zero_volume = 1;
            if( zero_mode_dimensionality == 1 )
            {
                zero_volume = zero_mode_length[0];
            }
            else if( zero_mode_dimensionality == 2 )
            {
                scalar area_factor = ( bravais_vectors[0].normalized().cross(bravais_vectors[1].normalized()) ).norm();
                zero_volume = zero_mode_length[0]*zero_mode_length[1] * area_factor;
            }
            else if( zero_mode_dimensionality == 3 )
            {
                scalar volume_factor = std::abs( (bravais_vectors[0].normalized().cross(bravais_vectors[1].normalized()) ).dot(
                    bravais_vectors[2].normalized()) );
                zero_volume = zero_mode_length[0]*zero_mode_length[1]*zero_mode_length[2] *  volume_factor;
            }

            std::cerr << "ZV zero mode dimensionality = " << zero_mode_dimensionality << std::endl;
            std::cerr << "ZV         zero mode length = " << zero_mode_length.transpose() << std::endl;
            std::cerr << "ZV = " << zero_volume << std::endl;

            // Return
            return zero_volume;
        }


        void Calculate_Perpendicular_Velocity(const vectorfield & spins, const scalarfield & mu_s, const MatrixX & hessian,
            const MatrixX & basis, const MatrixX & eigenbasis, VectorX & perpendicular_velocity)
        {
            int nos = spins.size();

            std::cerr << "  Calculate_Perpendicular_Velocity: calculate velocity matrix" << std::endl;

            // Calculate the velocity matrix in the 3N-basis
            MatrixX velocity(3*nos, 3*nos);
            Calculate_Dynamical_Matrix(spins, mu_s, hessian, velocity);

            std::cerr << "  Calculate_Perpendicular_Velocity: project velocity matrix" << std::endl;

            // Project the velocity matrix into the 2N tangent space
            MatrixX velocity_projected(2*nos, 2*nos);
            velocity_projected = basis.transpose()*velocity*basis;

            std::cerr << "  Calculate_Perpendicular_Velocity: calculate a" << std::endl;

            // QUESTION: is this maybe just eigenbasis^T * velocity_projected * eigenbasis ?
            // Something
            perpendicular_velocity = eigenbasis.col(0).transpose() * (velocity_projected*eigenbasis);

            // std::cerr << "  Calculate_Perpendicular_Velocity: sorting" << std::endl;
            // std::sort(perpendicular_velocity.data(),perpendicular_velocity.data()+perpendicular_velocity.size());

            for (int i=0; i<10; ++i)
                std::cerr << fmt::format("  a[{}] = {}", i, perpendicular_velocity[i]) << std::endl;
            // std::cerr << "without units:" << std::endl;
            // for (int i=0; i<10; ++i)
            //     std::cerr << "  a[" << i << "] = " << perpendicular_velocity[i]/C::mu_B << std::endl;
        }


        void Calculate_Dynamical_Matrix(const vectorfield & spins, const scalarfield & mu_s, const MatrixX & hessian, MatrixX & velocity)
        {
            velocity.setZero();
            int nos = spins.size();

            for (int i=0; i < nos; ++i)
            {
                Vector3 beff{0, 0, 0};

                for (int j=0; j < nos; ++j)
                {
                    velocity(3*i, 3*j)     = spins[i][1]*hessian(3*i+2,3*j)   - spins[i][2]*hessian(3*i+1,3*j);
                    velocity(3*i, 3*j+1)   = spins[i][1]*hessian(3*i+2,3*j+1) - spins[i][2]*hessian(3*i+1,3*j+1);
                    velocity(3*i, 3*j+2)   = spins[i][1]*hessian(3*i+2,3*j+2) - spins[i][2]*hessian(3*i+1,3*j+2);

                    velocity(3*i+1, 3*j)   = spins[i][2]*hessian(3*i,3*j)     - spins[i][0]*hessian(3*i+2,3*j);
                    velocity(3*i+1, 3*j+1) = spins[i][2]*hessian(3*i,3*j+1)   - spins[i][0]*hessian(3*i+2,3*j+1);
                    velocity(3*i+1, 3*j+2) = spins[i][2]*hessian(3*i,3*j+2)   - spins[i][0]*hessian(3*i+2,3*j+2);

                    velocity(3*i+2, 3*j)   = spins[i][0]*hessian(3*i+1,3*j)   - spins[i][1]*hessian(3*i,3*j);
                    velocity(3*i+2, 3*j+1) = spins[i][0]*hessian(3*i+1,3*j+1) - spins[i][1]*hessian(3*i,3*j+1);
                    velocity(3*i+2, 3*j+2) = spins[i][0]*hessian(3*i+1,3*j+2) - spins[i][1]*hessian(3*i,3*j+2);

                    beff -= hessian.block<3, 3>(3*i, 3*j) * spins[j];
                }

                velocity(3*i,   3*i+1) -= beff[2];
                velocity(3*i,   3*i+2) += beff[1];
                velocity(3*i+1, 3*i)   += beff[2];
                velocity(3*i+1, 3*i+2) -= beff[0];
                velocity(3*i+2, 3*i)   -= beff[1];
                velocity(3*i+2, 3*i+1) += beff[0];

                velocity.row(3*i)   /= mu_s[i];
                velocity.row(3*i+1) /= mu_s[i];
                velocity.row(3*i+2) /= mu_s[i];
            }
        }


        void hessian_bordered_3N(const vectorfield & image, const vectorfield & gradient, const MatrixX & hessian, MatrixX & hessian_out)
        {
            // Calculates a 3Nx3N matrix in the bordered Hessian approach and transforms it into the tangent basis,
            // making the result a 2Nx2N matrix. The bordered Hessian's Lagrange multipliers assume a local extremum.

            int nos = image.size();
            hessian_out = hessian;

            VectorX lambda(nos);
            for (int i=0; i<nos; ++i)
                lambda[i] = image[i].normalized().dot(gradient[i]);

            for (int i=0; i<nos; ++i)
            {
                for (int j=0; j<3; ++j)
                {
                    hessian_out(3*i+j,3*i+j) -= lambda[i];
                }
            }
        }

        // NOTE WE ASSUME A SELFADJOINT MATRIX
        void Eigen_Decomposition(const MatrixX & matrix, VectorX & evalues, MatrixX & evectors)
        {
            // Create a Spectra solver
            Eigen::SelfAdjointEigenSolver<MatrixX> matrix_solver(matrix);

            evalues = matrix_solver.eigenvalues().real();
            evectors = matrix_solver.eigenvectors().real();
        }

        void Eigen_Decomposition_Spectra(int nos, const MatrixX & matrix, VectorX & evalues, MatrixX & evectors, int n_decompose=1)
        {
            int n_steps = std::max(2, nos);

            //		Create a Spectra solver
            Spectra::DenseGenMatProd<scalar> op(matrix);
            Spectra::GenEigsSolver< scalar, Spectra::SMALLEST_REAL, Spectra::DenseGenMatProd<scalar> > matrix_spectrum(&op, n_decompose, n_steps);
            matrix_spectrum.init();

            //		Compute the specified spectrum
            int nconv = matrix_spectrum.compute();

            if (matrix_spectrum.info() == Spectra::SUCCESSFUL)
            {
                evalues = matrix_spectrum.eigenvalues().real();
                evectors = matrix_spectrum.eigenvectors().real();
                // Eigen::Ref<VectorX> evec = evectors.col(0);
            }
            else
            {
                Log(Utility::Log_Level::Error, Utility::Log_Sender::All, "Failed to calculate eigenvectors of the Matrix!");
                evalues.setZero();
                evectors.setZero();
            }
        }

        void Geodesic_Eigen_Decomposition(const vectorfield & image, const vectorfield & gradient, const MatrixX & hessian,
            MatrixX & hessian_geodesic_3N, MatrixX & hessian_geodesic_2N, VectorX & eigenvalues, MatrixX & eigenvectors)
        {
            std::cerr << "---------- Geodesic Eigen Decomposition" << std::endl;

            int nos = image.size();

            // Calculate geodesic Hessian in 3N-representation
            hessian_geodesic_3N = MatrixX::Zero(3*nos, 3*nos);
            hessian_bordered_3N(image, gradient, hessian, hessian_geodesic_3N);

            // Transform into geodesic Hessian
            std::cerr << "    Transforming Hessian into geodesic Hessian..." << std::endl;
            hessian_geodesic_2N = MatrixX::Zero(2*nos, 2*nos);
            // Manifoldmath::hessian_bordered(image, gradient, hessian, hessian_geodesic_2N);
            // Manifoldmath::hessian_projected(image, gradient, hessian, hessian_geodesic_2N);
            // Manifoldmath::hessian_weingarten(image, gradient, hessian, hessian_geodesic_2N);
            // Manifoldmath::hessian_spherical(image, gradient, hessian, hessian_geodesic_2N);
            // Manifoldmath::hessian_covariant(image, gradient, hessian, hessian_geodesic_2N);

            // Do this manually
            MatrixX basis = MatrixX::Zero(3*nos, 2*nos);
            Manifoldmath::tangent_basis_spherical(image, basis);
            // Manifoldmath::tangent_basis(image, basis);
            hessian_geodesic_2N = basis.transpose() * hessian_geodesic_3N * basis;


            // Calculate full eigenspectrum
            std::cerr << "    Calculation of full eigenspectrum..." << std::endl;
            // std::cerr << hessian_geodesic_2N.cols() << "   " << hessian_geodesic_2N.rows() << std::endl;
            eigenvalues = VectorX::Zero(2*nos);
            eigenvectors = MatrixX::Zero(2*nos, 2*nos);
            Eigen_Decomposition(hessian_geodesic_2N, eigenvalues, eigenvectors);
            // Eigen_Decomposition_Spectra(hessian_geodesic_2N, eigenvalues, eigenvectors);

            std::cerr << "---------- Geodesic Eigen Decomposition Done" << std::endl;
        }


    }// end namespace HTST
}// end namespace Engine