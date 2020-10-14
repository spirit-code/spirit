#ifndef SPIRIT_SKIP_HTST

#include <engine/HTST.hpp>
#include <engine/Sparse_HTST.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Manifoldmath.hpp>
#include <engine/Hamiltonian_Heisenberg.hpp>
#include <utility/Constants.hpp>
#include <utility/Logging.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <GenEigsSolver.h>  // Also includes <MatOp/DenseGenMatProd.h>
#include <SymEigsSolver.h>
#include <MatOp/SparseSymMatProd.h>

#include <GenEigsRealShiftSolver.h>

#include <fmt/format.h>
#include <fmt/ostream.h>

namespace C = Utility::Constants;

namespace Engine
{
    namespace Sparse_HTST
    {

        void Sparse_Get_Lowest_Eigenvector(const SpMatrixX & matrix, int nos, scalar & lowest_evalue, VectorX & lowest_evec)
        {
            VectorX evalues;
            MatrixX evectors;

            int n_steps = std::max(2, nos);

            //  Create a Spectra solver
            Spectra::SparseSymMatProd<scalar> op(matrix);
            Spectra::SymEigsSolver< scalar, Spectra::SMALLEST_ALGE, Spectra::SparseSymMatProd<scalar> > matrix_spectrum(&op, 1, n_steps);

            matrix_spectrum.init();
            int nconv = matrix_spectrum.compute();

            if (matrix_spectrum.info() == Spectra::SUCCESSFUL)
            {
                evalues = matrix_spectrum.eigenvalues().real();
                evectors = matrix_spectrum.eigenvectors().real();
            } else {
                Log(Utility::Log_Level::All, Utility::Log_Sender::HTST, "    Failed to calculate lowest eigenmode. Aborting!");
                return;
            }

            lowest_evalue = evalues[0];
            lowest_evec = evectors.col(0);
        }

        void Sparse_Get_Lowest_Eigenvector_VP(const SpMatrixX & matrix, int nos, const VectorX & init, scalar & lowest_evalue, VectorX & lowest_evec)
        {
            auto & x = lowest_evec;
            x = init;
            x.normalize();

            scalar tol = 1e-6;
            scalar cur = 2 * tol;
            scalar m = 0.01;
            scalar step_size = 1e-4;
            int n_iter = 0;

            VectorX gradient = VectorX::Zero(2*nos);
            VectorX gradient_prev = VectorX::Zero(2*nos);
            VectorX velocity = VectorX::Zero(2*nos);

            VectorX mx; // temporary for matrix * x
            scalar proj; // temporary for gradient * x
            scalar fnorm2=2*tol*tol, ratio;

            while (std::sqrt(fnorm2) > tol)
            {
                // Compute gradient of Rayleigh quotient
                mx = matrix * x;
                gradient = 2 * mx;
                proj = gradient.dot(x);
                gradient -= proj * x;
                // std::cout << "proj " << proj << "\n";
                // std::cout << "x " << x.head(5).transpose() << "\n";
                // std::cout << "gradient " << gradient.head(5).transpose() << "\n";


                velocity = 0.5 * (gradient + gradient_prev) / m;
                fnorm2 = gradient.squaredNorm();

                proj = velocity.dot(gradient);
                ratio = proj/fnorm2;

                if (proj<=0)
                    velocity.setZero();
                else 
                    velocity = gradient*ratio;

                // Update x and renormalize
                x -= step_size * velocity + 0.5/m * step_size * gradient;
                x.normalize();

                // Update prev gradient
                gradient_prev = gradient;
                // std::cout << "fnorm2 " << fnorm2 << "\n";
                // std::cout << "proj " << proj << "\n";

                // Increment n_iter
                lowest_evalue = x.dot(matrix * x);
                // std::cout << "lowest_evalue " << lowest_evalue << "\n";
                n_iter++;
            }

            // Compute the eigenvalue
            lowest_evalue = x.dot(matrix * x);
            std::cout << "n_iter " << n_iter << "\n";
            std::cout << "evalue " << lowest_evalue << "\n";
        }

        // Note the two images should correspond to one minimum and one saddle point
        // Non-extremal images may yield incorrect Hessians and thus incorrect results
        void Calculate(Data::HTST_Info & htst_info)
        {
            Log(Utility::Log_Level::All, Utility::Log_Sender::HTST, "Sparse Prefactor calculation");
            bool lowest_mode_spectra=false;
            htst_info.sparse = true;
            htst_info.n_eigenmodes_keep = 0;

            const scalar epsilon = 1e-4;
            const scalar epsilon_force = 1e-8;

            auto& image_minimum = *htst_info.minimum->spins;
            auto& image_sp      = *htst_info.saddle_point->spins;

            int nos = image_minimum.size();

            Log(Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Saving NO eigenvectors.");

            vectorfield force_tmp(nos, {0,0,0});
            std::vector<std::string> block;

            // TODO
            bool is_afm = false;

            // The gradient (unprojected)
            Log(Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Evaluation of the gradient at the initial configuration...");
            vectorfield gradient_minimum(nos, {0,0,0});
            htst_info.minimum->hamiltonian->Gradient(image_minimum, gradient_minimum);

            // Check if the configuration is actually an extremum
            Log(Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Checking if initial configuration is an extremum...");
            Vectormath::set_c_a(1, gradient_minimum, force_tmp);
            Manifoldmath::project_tangential(force_tmp, image_minimum);
            scalar fmax_minimum = Vectormath::max_norm(force_tmp);
            if( fmax_minimum > epsilon_force )
            {
                Log(Utility::Log_Level::Error, Utility::Log_Sender::All, fmt::format(
                    "HTST: the initial configuration is not a converged minimum, its max. torque is above the threshold ({} > {})!", fmax_minimum, epsilon_force ));
                return;
            }

            // The gradient (unprojected)
            Log(Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Evaluation of the gradient at the transition configuration...");
            vectorfield gradient_sp(nos, {0,0,0});
            htst_info.saddle_point->hamiltonian->Gradient(image_sp, gradient_sp);

            // Check if the configuration is actually an extremum
            Log(Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Checking if transition configuration is an extremum...");
            Vectormath::set_c_a(1, gradient_sp, force_tmp);
            Manifoldmath::project_tangential(force_tmp, image_sp);
            scalar fmax_sp = Vectormath::max_norm(force_tmp);
            if( fmax_sp > epsilon_force )
            {
                Log(Utility::Log_Level::Error, Utility::Log_Sender::All, fmt::format(
                    "HTST: the transition configuration is not a converged saddle point, its max. torque is above the threshold ({} > {})!", fmax_sp, epsilon_force ));
                return;
            }

            ////////////////////////////////////////////////////////////////////////
            // Saddle point
            {
                Log(Utility::Log_Level::Info, Utility::Log_Sender::HTST, "Calculation for the Saddle Point");

                Log(Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Evaluate tangent basis ...");
                SpMatrixX tangent_basis = SpMatrixX(3*nos, 2*nos);
                Manifoldmath::sparse_tangent_basis_spherical(image_sp, tangent_basis);

                // Evaluation of the Hessian...
                Log(Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Evaluate the Hessian...");
                SpMatrixX sparse_hessian_sp(3*nos, 3*nos);
                htst_info.saddle_point->hamiltonian->Sparse_Hessian(image_sp, sparse_hessian_sp);

                // Transform into geodesic Hessian
                Log(Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Transform Hessian into geodesic Hessian...");
                SpMatrixX sparse_hessian_sp_geodesic_3N(3*nos, 3*nos);
                sparse_hessian_bordered_3N(image_sp, gradient_sp, sparse_hessian_sp, sparse_hessian_sp_geodesic_3N);
                SpMatrixX sparse_hessian_sp_geodesic_2N = tangent_basis.transpose() * sparse_hessian_sp_geodesic_3N * tangent_basis;

                Log(Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Evaluate lowest eigenmode of the Hessian...");
                VectorX lowest_evector(2*nos);
                scalar lowest_evalue;

                if(lowest_mode_spectra)
                {
                    Sparse_Get_Lowest_Eigenvector(sparse_hessian_sp_geodesic_2N, nos, lowest_evalue, lowest_evector);
                } else {
                    VectorX init = VectorX::Random(2*nos);
                    init[0] = 1.0;
                    Sparse_Get_Lowest_Eigenvector_VP(sparse_hessian_sp_geodesic_2N, nos, init, lowest_evalue, lowest_evector);
                };
                Log(Utility::Log_Level::Info, Utility::Log_Sender::HTST, fmt::format("        Lowest eigenvalue: {}", lowest_evalue));
                if(2*nos>=4)
                    Log(Utility::Log_Level::Info, Utility::Log_Sender::HTST, fmt::format("        Lowest eigenvector: {}, {}, {}, ... {}", lowest_evector[0], lowest_evector[1], lowest_evector[2], lowest_evector[2*nos-1]));

                // Check if lowest eigenvalue < 0 (else it's not a SP)
                Log(Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Check if actually a saddle point...");
                if( lowest_evalue > -epsilon )
                {
                    Log(Utility::Log_Level::Error, Utility::Log_Sender::All, fmt::format(
                        "HTST: the transition configuration is not a saddle point, its lowest eigenvalue is above the threshold ({} > {})!", htst_info.eigenvalues_sp[0], -epsilon ));
                    return;
                }

                Log(Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Sparse LU Decomposition of geodesic Hessian...");
                Eigen::SparseLU<SpMatrixX, Eigen::COLAMDOrdering<int> > solver;
                solver.analyzePattern(sparse_hessian_sp_geodesic_2N);
                solver.factorize(sparse_hessian_sp_geodesic_2N);
                htst_info.det_sp = solver.logAbsDeterminant() - std::log(-lowest_evalue);

                // Perpendicular velocity
                Log(Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Calculate dynamical contribution");

                Log(Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Evaluate the dynamical matrix");
                SpMatrixX velocity(3*nos, 3*nos);
                Sparse_Calculate_Dynamical_Matrix(image_sp, htst_info.saddle_point->geometry->mu_s, sparse_hessian_sp_geodesic_3N, velocity);
                SpMatrixX projected_velocity = tangent_basis.transpose() * velocity * tangent_basis;

                Log(Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Solving H^-1 V q_1 ...");
                VectorX x(2*nos);
                x = solver.solve(projected_velocity.transpose() * lowest_evector);
                htst_info.s = std::sqrt(lowest_evector.transpose() * projected_velocity * x );
            }
            // End saddle point
            ////////////////////////////////////////////////////////////////////////

            int n_zero_modes_sp = 0;

            // TODO
            htst_info.volume_sp = 1;

            ////////////////////////////////////////////////////////////////////////
            // Initial state minimum
            {
                Log(Utility::Log_Level::Info, Utility::Log_Sender::HTST, "Calculation for the Minimum");

                Log(Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Evaluate tangent basis ...");
                SpMatrixX tangent_basis = SpMatrixX(3*nos, 2*nos);
                Manifoldmath::sparse_tangent_basis_spherical(image_minimum, tangent_basis);

                // Evaluation of the Hessian...
                Log(Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Evaluate the Hessian...");
                SpMatrixX sparse_hessian_minimum = SpMatrixX(3*nos,3*nos);
                htst_info.minimum->hamiltonian->Sparse_Hessian(image_minimum, sparse_hessian_minimum);

                // Transform into geodesic Hessian
                Log(Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Transforming Hessian into geodesic Hessian...");
                SpMatrixX sparse_hessian_geodesic_min_3N = SpMatrixX(3*nos, 3*nos);
                sparse_hessian_bordered_3N(image_minimum, gradient_minimum, sparse_hessian_minimum, sparse_hessian_geodesic_min_3N);
                SpMatrixX sparse_hessian_geodesic_min_2N = tangent_basis.transpose() * sparse_hessian_geodesic_min_3N * tangent_basis;

                Log(Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Sparse LU Decomposition of geodesic Hessian...");
                Eigen::SparseLU<SpMatrixX, Eigen::COLAMDOrdering<int> > solver;
                solver.analyzePattern(sparse_hessian_geodesic_min_2N);
                solver.factorize(sparse_hessian_geodesic_min_2N);
                htst_info.det_min = solver.logAbsDeterminant();
            }
            // End initial state minimum
            ////////////////////////////////////////////////////////////////////////

            // Checking for zero modes at the minimum...
            Log(Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Checking for zero modes at the minimum...");
            int n_zero_modes_minimum = 0;

            //TODO
            htst_info.volume_min = 1;


            ////////////////////////////////////////////////////////////////////////
            // Calculation of the prefactor...
            Log(Utility::Log_Level::Info, Utility::Log_Sender::HTST, "Calculating prefactor...");

            // Calculate the exponent for the temperature-dependence of the prefactor
            //      The exponent depends on the number of zero modes at the different states
            htst_info.temperature_exponent = 0.5 * (n_zero_modes_minimum - n_zero_modes_sp);

            // Calculate "me"
            htst_info.me = std::pow(2*C::Pi * C::k_B, htst_info.temperature_exponent);

            // Calculate Omega_0, i.e. the entropy contribution
            htst_info.Omega_0 = 1;
            htst_info.Omega_0 = std::sqrt( std::exp(htst_info.det_min - htst_info.det_sp) );

            // Calculate the prefactor
            htst_info.prefactor_dynamical = htst_info.me * htst_info.volume_sp / htst_info.volume_min * htst_info.s;
            htst_info.prefactor = C::g_e / (C::hbar * 1e-12) * htst_info.Omega_0 * htst_info.prefactor_dynamical / ( 2*C::Pi );

            Log.SendBlock(Utility::Log_Level::All, Utility::Log_Sender::HTST,
                {
                    "---- Prefactor calculation successful!",
                    fmt::format("exponent      = {:^20e}", htst_info.temperature_exponent),
                    fmt::format("me            = {:^20e}", htst_info.me),
                    fmt::format("m = Omega_0   = {:^20e}", htst_info.Omega_0),
                    fmt::format("s             = {:^20e}", htst_info.s),
                    fmt::format("volume_sp     = {:^20e}", htst_info.volume_sp),
                    fmt::format("volume_min    = {:^20e}", htst_info.volume_min),
                    fmt::format("log |det_min| = {:^20e}", htst_info.det_min),
                    fmt::format("log |det_sp|  = {:^20e}", htst_info.det_sp),
                    fmt::format("hbar[meV*s]   = {:^20e}", C::hbar*1e-12),
                    fmt::format("v = dynamical prefactor = {:^20e}", htst_info.prefactor_dynamical),
                    fmt::format("prefactor               = {:^20e}", htst_info.prefactor)
                }, -1, -1);
        }

         void Sparse_Calculate_Dynamical_Matrix(const vectorfield & spins, const scalarfield & mu_s, const SpMatrixX & hessian, SpMatrixX & velocity)
        {
            constexpr scalar epsilon = 1e-10;
            int nos = spins.size();

            typedef Eigen::Triplet<scalar> T;
            std::vector<T> tripletList;
            tripletList.reserve(3 * nos);

            // TODO not very efficient. Iterate directly over non-zero entries of Hessian, how to best figure them out?
            scalar temp = 0;
            for (int i=0; i < nos; ++i)
            {
                Vector3 beff{0, 0, 0};
                for (int j=0; j < nos; ++j)
                {
                    // TODO: Some of these if checks are superfluous 
                    temp = ( spins[i][1]*hessian.coeff(3*i+2,3*j)   - spins[i][2]*hessian.coeff(3*i+1,3*j) ) / mu_s[i];
                    if( std::abs(temp) > epsilon )
                    {
                        tripletList.push_back(T(3*i, 3*j, temp));
                    }

                    temp = ( spins[i][1]*hessian.coeff(3*i+2,3*j+1) - spins[i][2]*hessian.coeff(3*i+1,3*j+1) ) / mu_s[i];
                    if(std::abs(temp) > epsilon)
                    {
                        tripletList.push_back(T(3*i, 3*j+1, temp));
                    }

                    temp = ( spins[i][1]*hessian.coeff(3*i+2,3*j+2) - spins[i][2]*hessian.coeff(3*i+1,3*j+2) ) / mu_s[i];
                    if(std::abs(temp) > epsilon)
                    {
                        tripletList.push_back(T(3*i, 3*j+2, temp));
                    }

                    // ---

                    temp = ( spins[i][2]*hessian.coeff(3*i,3*j) - spins[i][0]*hessian.coeff(3*i+2,3*j) ) / mu_s[i];
                    if(std::abs(temp) > epsilon)
                    {
                        tripletList.push_back(T(3*i+1, 3*j, temp));
                    }

                    temp = ( spins[i][2]*hessian.coeff(3*i,3*j+1) - spins[i][0]*hessian.coeff(3*i+2,3*j+1) ) / mu_s[i];
                    if (std::abs(temp) > epsilon)
                    {
                        tripletList.push_back(T(3*i+1, 3*j+1, temp));
                    }

                    temp = ( spins[i][2]*hessian.coeff(3*i,3*j+2) - spins[i][0]*hessian.coeff(3*i+2,3*j+2) ) / mu_s[i];
                    if (std::abs(temp) > epsilon)
                    {
                        tripletList.push_back(T(3*i+1, 3*j+2, temp));
                    }

                    // ---

                    temp = ( spins[i][0]*hessian.coeff(3*i+1,3*j) - spins[i][1]*hessian.coeff(3*i,3*j) ) / mu_s[i];
                    if (std::abs(temp) > epsilon)
                    {
                        tripletList.push_back(T(3*i+2, 3*j, temp));
                    }

                    temp = ( spins[i][0]*hessian.coeff(3*i+1,3*j+1) - spins[i][1]*hessian.coeff(3*i,3*j+1) ) / mu_s[i];
                    if (std::abs(temp) > epsilon)
                    {
                        tripletList.push_back(T(3*i+2, 3*j+1, temp));
                    }

                    temp = ( spins[i][0]*hessian.coeff(3*i+1,3*j+2) - spins[i][1]*hessian.coeff(3*i,3*j+2) ) / mu_s[i];
                    if (std::abs(temp) > epsilon)
                    {
                        tripletList.push_back(T(3*i+2, 3*j+2, temp));
                    }
                    beff -= hessian.block(3*i, 3*j, 3, 3) * spins[j] / mu_s[i];
                }

                tripletList.push_back( T(3*i,   3*i+1, -beff[2]));
                tripletList.push_back( T(3*i,   3*i+2,  beff[1]));
                tripletList.push_back( T(3*i+1, 3*i,    beff[2]));
                tripletList.push_back( T(3*i+1, 3*i+2, -beff[0]));
                tripletList.push_back( T(3*i+2, 3*i,   -beff[1]));
                tripletList.push_back( T(3*i+2, 3*i+1,  beff[0]));

                velocity.setFromTriplets(tripletList.begin(), tripletList.end());
            }
        }

        void sparse_hessian_bordered_3N(const vectorfield & image, const vectorfield & gradient, const SpMatrixX & hessian, SpMatrixX & hessian_out)
        {
            // Calculates a 3Nx3N matrix in the bordered Hessian approach and transforms it into the tangent basis,
            // making the result a 2Nx2N matrix. The bordered Hessian's Lagrange multipliers assume a local extremum.

            int nos = image.size();
            VectorX lambda(nos);
            for (int i=0; i<nos; ++i)
                lambda[i] = image[i].normalized().dot(gradient[i]);

            // Construct hessian_out
            typedef Eigen::Triplet<scalar> T;
            std::vector<T> tripletList;
            tripletList.reserve( hessian.nonZeros() + 3*nos );

            // Iterate over non zero entries of hesiian
            for (int k=0; k<hessian.outerSize(); ++k)
            {
                for (SpMatrixX::InnerIterator it(hessian,k); it; ++it)
                {
                    tripletList.push_back( T(it.row(), it.col(), it.value() ) );
                }
                int j = k % 3;
                int i = (k - j) / 3;
                tripletList.push_back( T(k,k, -lambda[i]) ); // Correction to the diagonal
            }
            hessian_out.setFromTriplets(tripletList.begin(), tripletList.end());
        }

        // NOTE WE ASSUME A SELFADJOINT MATRIX
        void Sparse_Eigen_Decomposition(const SpMatrixX & matrix, VectorX & evalues, MatrixX & evectors)
        {
            // Create a Spectra solver
            Eigen::SelfAdjointEigenSolver<SpMatrixX> matrix_solver(matrix);
            evalues = matrix_solver.eigenvalues().real();
            evectors = matrix_solver.eigenvectors().real();
        }

        void Sparse_Geodesic_Eigen_Decomposition(const vectorfield & image, const vectorfield & gradient, const SpMatrixX & hessian,
            SpMatrixX & hessian_geodesic_3N, SpMatrixX & hessian_geodesic_2N, SpMatrixX & tangent_basis, VectorX & eigenvalues, MatrixX & eigenvectors)
        {
            Log(Utility::Log_Level::Info, Utility::Log_Sender::HTST, "---------- Sparse Geodesic Eigen Decomposition");

            int nos = image.size();

            // Calculate geodesic Hessian in 3N-representation
            hessian_geodesic_3N = SpMatrixX(3*nos, 3*nos);
            sparse_hessian_bordered_3N(image, gradient, hessian, hessian_geodesic_3N);

            // Transform into geodesic Hessian
            Log(Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Transforming Hessian into geodesic Hessian...");
            hessian_geodesic_2N = SpMatrixX(2*nos, 2*nos);
            hessian_geodesic_2N = tangent_basis.transpose() * hessian_geodesic_3N * tangent_basis;

            // Calculate full eigenspectrum
            Log(Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Calculation of full eigenspectrum..." );

            eigenvalues = VectorX::Zero(2*nos);
            eigenvectors = MatrixX::Zero(2*nos, 2*nos);

            Sparse_Eigen_Decomposition(hessian_geodesic_2N, eigenvalues, eigenvectors);

            Log(Utility::Log_Level::Info, Utility::Log_Sender::HTST, "---------- Sparse Geodesic Eigen Decomposition Done");
        }

    }
}
#endif