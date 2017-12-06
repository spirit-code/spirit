#include <engine/HTST.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Manifoldmath.hpp>
#include <data/Spin_System.hpp>
#include <utility/Constants.hpp>
#include <utility/Logging.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
// #include <Eigen/Array>
#include <Eigen/Eigenvalues>
//#include <unsupported/Eigen/CXX11/Tensor>
#include <GenEigsSolver.h>  // Also includes <MatOp/DenseGenMatProd.h>
#include <GenEigsRealShiftSolver.h>

#include <iostream>

namespace Engine
{
    namespace HTST
    {
        // Note the two images should correspond to one minimum and one saddle point
        // Non-extremal images may yield incorrect Hessians and thus incorrect results
        scalar Get_Prefactor(std::shared_ptr<Data::Spin_System> system_minimum, std::shared_ptr<Data::Spin_System> system_sp)
        {
            std::cerr << "Getting Prefactor" << std::endl;

            bool is_afm = false;

            scalar epsilon = 1e-4;
            
            ////////////////////////////////////////////////////////////////////////
            // Initial state minimum
            std::cerr << std::endl << "Calculation for the Minimum" << std::endl;

            // The gradient (unprojected)
            std::cerr << "    Evaluation of the gradient..." << std::endl;
            auto& image_minimum = *system_minimum->spins;
            int nos = image_minimum.size();
            vectorfield gradient_minimum(nos, {0,0,0});
            system_minimum->hamiltonian->Gradient(image_minimum, gradient_minimum);

            // Evaluation of the Hessian...
            MatrixX hessian_minimum = MatrixX::Zero(3*nos,3*nos);
            std::cerr << "    Evaluation of the Hessian..." << std::endl;
            system_minimum->hamiltonian->Hessian(image_minimum, hessian_minimum);

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
                std::cerr << "ew[" << i << "]=" << eigenvalues_minimum[i] << std::endl;

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
                std::cerr << "ZERO MODES AT MINIMUM (N=" << n_zero_modes_minimum << ")" << std::endl;
                
                if (is_afm)
                    volume_minimum = Calculate_Zero_Volume(system_minimum);
                else
                    volume_minimum = Calculate_Zero_Volume(system_minimum);
            }

            ////////////////////////////////////////////////////////////////////////
            // Saddle point
            std::cerr << std::endl << "Calculation for the Saddle Point" << std::endl;

            // The gradient (unprojected)
            std::cerr << "    Evaluation of the gradient..." << std::endl;
            auto& image_sp = *system_sp->spins;
            vectorfield gradient_sp(nos, {0,0,0});
            system_sp->hamiltonian->Gradient(image_sp, gradient_sp);

            // Evaluation of the Hessian...
            MatrixX hessian_sp = MatrixX::Zero(3*nos,3*nos);
            std::cerr << "    Evaluation of the Hessian..." << std::endl;
            system_sp->hamiltonian->Hessian(image_sp, hessian_sp);

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
                std::cerr << "ew[" << i << "]=" << eigenvalues_sp[i] << std::endl;

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
                    std::cerr << "HIGHER ORDER SADDLE POINT (N=" << n_negative << ")" << std::endl;
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
                std::cerr << "ZERO MODES AT SADDLE POINT (N=" << n_zero_modes_sp << ")" << std::endl;

                if (is_afm)
                    volume_sp = Calculate_Zero_Volume(system_sp);
                else
                    volume_sp = Calculate_Zero_Volume(system_sp);
            }

            ////////////////////////////////////////////////////////////////////////
            std::cerr << "Calculating 'a' factors..." << std::endl;
            // Calculation of the 'a' parameters...
            VectorX a_sp(2*nos);
            MatrixX basis_sp(3*nos, 2*nos);
            Manifoldmath::tangent_basis_spherical(image_sp, basis_sp);
            // Manifoldmath::tangent_basis(image_sp, basis_sp);
            // TODO
            // Calculate_a_2N(image_sp, hessian_geodesic_sp_2N, basis_sp, eigenvectors_sp, a_sp);
            Calculate_a(image_sp, hessian_geodesic_sp_3N, basis_sp, eigenvectors_sp, a_sp);
            // QUESTION: is scaling a_sp with mub/mry necessary?
            
            ////////////////////////////////////////////////////////////////////////
            // Calculation of the prefactor...
            std::cerr << "Calculating prefactor..." << std::endl;
            scalar prefactor, exponent;
            Calculate_Prefactor(nos, n_zero_modes_minimum, n_zero_modes_sp, volume_minimum, volume_sp, eigenvalues_minimum, eigenvalues_sp, a_sp,
                prefactor, exponent);

            ////////////////////////////////////////////////////////////////////////
            // Return
            return prefactor;
        }

        scalar Calculate_Zero_Volume(const std::shared_ptr<Data::Spin_System> system)
        {
            int   nos      = system->geometry->nos;
            auto& n_cells  = system->geometry->n_cells;
            auto& spins    = *system->spins;
            auto& spin_pos = system->geometry->spin_pos;
            auto& bc       = system->hamiltonian->boundary_conditions;
            
            // QUESTION: what 3x3 basis should this be? The translations? The basis of the unit cells? -> translation vectors
            MatrixX basis = MatrixX::Identity(3, 3);

            scalar volume = 1;

            // l_zm is the zero mode length
            // kk is ??
            Vector3 kk{0,0,0}, l_zm{0,0,0};
            // what is ll ?? (used to be ll1, ll2, ll3)
            Vector3 ll{0,0,0};

            // auxiliary variables
            Vector3 test_pos;
            MatrixX auxM(nos, 3);

            MatrixX auxb = basis;
            
            // QUESTION: what the heck is this? -> LU factorization
            // call dgetrf(3,3,auxb,3,ipiv,info)
            Eigen::ColPivHouseholderQR<MatrixX> lu_factorized(auxb);
            
            // Dimensionality of the zero mode
            int dim_zm = 0;
            for (int i=0; i<3; ++i)
            {
                // QUESTION: why only for periodical boundaries?
                if (bc[i])
                {
                    kk[dim_zm] = i;
                    
                    for (int j=0; j<nos; ++j)
                    {
                        test_pos = spin_pos[j] + basis.col(i); // QUESTION: is this the right order?

                        // QUESTION: what is this? -> solving system of linear equations
                        // call dgetrs('N',3,1,auxb,3,ipiv,v,3,info)
                        test_pos = lu_factorized.solve(test_pos);

                        for (int dim=0; dim<3; ++dim)
                        {
                            if (std::round(test_pos[dim]) > n_cells[dim]-1)
                                test_pos[dim] = 0;
                            else if (std::round(test_pos[dim]) < 0)
                                test_pos[dim] = n_cells[0]-1;
                            // auxv[dim] = v[dim];
                        }

                        // QUESTION: what is this? -> alpha*A*x + beta*y -> y
                        // call dgemv('N',3,3,1d0,basis,3,auxv,1,0d0,v,1)
                        test_pos = basis*test_pos;
                        
                        // Find the first position of test_pos inside the spin_pos field
                        int num = find_pos(test_pos, spin_pos);
                        if (num<0)
                        {
                            std::cerr << "WARNING: Failed to move system along zero mode" << std::endl;
                            return volume;
                        }
                        
                        auxM.row(num) = spins[j]; // QUESTION: is this the right order?
                    }
                    
                    for (int j=0; j<nos; ++j)
                    {
                        // QUESTION: is this simply the angle between spins?
                        // ll1 = calc_ang(emomM(:,j),auxM(:,j))
                        l_zm[dim_zm] += std::pow(Vectormath::angle(spins[j], auxM.row(j)), 2); // QUESTION: is this the right order?
                    }
                    l_zm[dim_zm] = std::sqrt(l_zm[dim_zm]);

                    // Increment zero mode dimensionality
                    ++dim_zm;
                }
            }

            // Calculate auxiliary basis?
            for (int i=0; i<3; ++i)
            {
                for (int j=0; j<3; ++j)
                {
                    auxb(i,j) = basis(i,j)/basis.row(j).norm();//norm_vec(3,basis(:,j));
                }
            }
            
            // Calculate the volume depending on the number of periodical boundaries
            if (dim_zm == 1)
                volume = l_zm[0];
            else if (dim_zm == 2)
            {
                ll[0] = auxb(1,kk[0])*auxb(2,kk[1])-auxb(2,kk[0])*auxb(1,kk[1]);
                ll[1] = auxb(0,kk[0])*auxb(2,kk[1])-auxb(2,kk[0])*auxb(0,kk[1]);
                ll[2] = auxb(0,kk[0])*auxb(1,kk[1])-auxb(1,kk[0])*auxb(0,kk[1]);
                volume = l_zm[0]*l_zm[1] * ll.norm();
            }
            else if (dim_zm == 3)
            {
                ll[0] = auxb(1,kk[0])*auxb(2,kk[1])-auxb(2,kk[0])*auxb(1,kk[1]);
                ll[1] = auxb(0,kk[0])*auxb(2,kk[1])-auxb(2,kk[0])*auxb(0,kk[1]);
                ll[2] = auxb(0,kk[0])*auxb(1,kk[1])-auxb(1,kk[0])*auxb(0,kk[1]);
                volume = l_zm[0]*l_zm[1]*l_zm[2]* std::abs( auxb.row(kk[2]).dot(ll) );
            }
            else
                volume = 1;

            std::cerr << "ZV zero mode dimensionality = " << dim_zm << std::endl;
            // std::cerr << "ZV                     auxb = " << std::endl << auxb << std::endl;
            std::cerr << "ZV         zero mode length = " << l_zm.transpose() << std::endl;
            std::cerr << "ZV                       ll = " << ll.transpose() << std::endl;
            std::cerr << "ZV                       kk = " << kk.transpose() << std::endl;
            std::cerr << "ZV = " << volume << std::endl;

            // Return
            return volume;
        }


        void Calculate_a(const vectorfield & spins, const MatrixX & hessian,
            const MatrixX & basis, const MatrixX & eigenbasis, VectorX & a)
        {
            int nos = spins.size();

            std::cerr << "  calculate_a: calculate velocity matrix" << std::endl;

            // Calculate the velocity matrix in the 3N-basis
            MatrixX velocity(3*nos, 3*nos);
            Calculate_Velocity_Matrix(spins, hessian, velocity);
            
            std::cerr << "  calculate_a: project velocity matrix" << std::endl;

            // Project the velocity matrix into the 2N tangent space
            MatrixX velocity_projected(2*nos, 2*nos);
            velocity_projected = basis.transpose()*velocity*basis;
            
            std::cerr << "  calculate_a: calculate a" << std::endl;

            // QUESTION: is this maybe just eigenbasis^T * velocity_projected * eigenbasis ?
            // Something
            a = eigenbasis.col(0).transpose() * (velocity_projected*eigenbasis);

            // std::cerr << "  calculate_a: sorting" << std::endl;
            // std::sort(a.data(),a.data()+a.size());

            // for (int i=0; i<10; ++i)
            //     std::cerr << "  a[" << i << "] = " << a[i] << std::endl;
        }


        void Calculate_Velocity_Matrix(const vectorfield & spins, const MatrixX & hessian, MatrixX & velocity)
        {
            velocity.setZero();
            int nos = spins.size();

            for (int i=0; i < nos; ++i)
            {
                Vector3 beff{0, 0, 0};
                
                for (int j=0; j < nos; ++j)
                {
                    // QUESTION: is there a nicer formula for this?
                    velocity(3*i, 3*j)     = 0.5 * (spins[i][1]*(hessian(3*j,  3*i+2)+hessian(3*i+2,3*j))   - spins[i][2]*(hessian(3*j,  3*i+1)+hessian(3*i+1,3*j)));
                    velocity(3*i, 3*j+1)   = 0.5 * (spins[i][1]*(hessian(3*j+1,3*i+2)+hessian(3*i+2,3*j+1)) - spins[i][2]*(hessian(3*j+1,3*i+1)+hessian(3*i+1,3*j+1)));
                    velocity(3*i, 3*j+2)   = 0.5 * (spins[i][1]*(hessian(3*j+2,3*i+2)+hessian(3*i+2,3*j+2)) - spins[i][2]*(hessian(3*j+2,3*i+1)+hessian(3*i+1,3*j+2)));

                    velocity(3*i+1, 3*j)   = 0.5 * (spins[i][2]*(hessian(3*j,  3*i)+hessian(3*i,3*j))       - spins[i][0]*(hessian(3*j,  3*i+2)+hessian(3*i+2,3*j)));
                    velocity(3*i+1, 3*j+1) = 0.5 * (spins[i][2]*(hessian(3*j+1,3*i)+hessian(3*i,3*j+1))     - spins[i][0]*(hessian(3*j+1,3*i+2)+hessian(3*i+2,3*j+1)));
                    velocity(3*i+1, 3*j+2) = 0.5 * (spins[i][2]*(hessian(3*j+2,3*i)+hessian(3*i,3*j+2))     - spins[i][0]*(hessian(3*j+2,3*i+2)+hessian(3*i+2,3*j+2)));

                    velocity(3*i+2, 3*j)   = 0.5 * (spins[i][0]*(hessian(3*j,  3*i+1)+hessian(3*i+1,3*j))   - spins[i][1]*(hessian(3*j,  3*i)+hessian(3*i,3*j)));
                    velocity(3*i+2, 3*j+1) = 0.5 * (spins[i][0]*(hessian(3*j+1,3*i+1)+hessian(3*i+1,3*j+1)) - spins[i][1]*(hessian(3*j+1,3*i)+hessian(3*i,3*j+1)));
                    velocity(3*i+2, 3*j+2) = 0.5 * (spins[i][0]*(hessian(3*j+2,3*i+1)+hessian(3*i+1,3*j+2)) - spins[i][1]*(hessian(3*j+2,3*i)+hessian(3*i,3*j+2)));

                    // ////////
                    // velocity(3*i, 3*j)     = spins[i][1]*hessian(3*i+2,3*j)   - spins[i][2]*hessian(3*i+1,3*j);
                    // velocity(3*i, 3*j+1)   = spins[i][1]*hessian(3*i+2,3*j+1) - spins[i][2]*hessian(3*i+1,3*j+1);
                    // velocity(3*i, 3*j+2)   = spins[i][1]*hessian(3*i+2,3*j+2) - spins[i][2]*hessian(3*i+1,3*j+2);

                    // velocity(3*i+1, 3*j)   = spins[i][2]*hessian(3*i,3*j)     - spins[i][0]*hessian(3*i+2,3*j);
                    // velocity(3*i+1, 3*j+1) = spins[i][2]*hessian(3*i,3*j+1)   - spins[i][0]*hessian(3*i+2,3*j+1);
                    // velocity(3*i+1, 3*j+2) = spins[i][2]*hessian(3*i,3*j+2)   - spins[i][0]*hessian(3*i+2,3*j+2);

                    // velocity(3*i+2, 3*j)   = spins[i][0]*hessian(3*i+1,3*j)   - spins[i][1]*hessian(3*i,3*j);
                    // velocity(3*i+2, 3*j+1) = spins[i][0]*hessian(3*i+1,3*j+1) - spins[i][1]*hessian(3*i,3*j+1);
                    // velocity(3*i+2, 3*j+2) = spins[i][0]*hessian(3*i+1,3*j+2) - spins[i][1]*hessian(3*i,3*j+2);
                    // ////////

                    beff -= (hessian.block<3, 3>(3*i, 3*j) + hessian.transpose().block<3, 3>(3*i, 3*j)) * spins[i];
                }

                beff = 0.5*beff;
                // if ( beff.norm() > 1e-7 && !(beff.normalized()).isApprox(spins[i].normalized()))
                //     std::cerr << "$$$$$$$$$$$$ " << i << ": " << beff.normalized().transpose() << " --- " << spins[i].normalized().transpose() << std::endl;

                velocity(3*i,   3*i+1) -= beff[2];
                velocity(3*i,   3*i+2) += beff[1];
                velocity(3*i+1, 3*i)   += beff[2];
                velocity(3*i+1, 3*i+2) -= beff[0];
                velocity(3*i+2, 3*i)   -= beff[1];
                velocity(3*i+2, 3*i+1) += beff[0];
            }
        }


        void Calculate_a_2N(const vectorfield & spins, const MatrixX & hessian,
            const MatrixX & basis, const MatrixX & eigenbasis, VectorX & a)
        {
            int nos = spins.size();
            // a = VectorX(2*nos);

            std::cerr << "  calculate_a: calculate dynamical matrix" << std::endl;

            // Calculate the dynamical matrix in the 2N-basis
            MatrixX dynamical(2*nos, 2*nos);
            Calculate_Dynamical_Matrix_2N(hessian, dynamical);
            
            std::cerr << "  calculate_a: calculate a" << std::endl;

            // QUESTION: is this maybe just eigenbasis^T * velocity_projected * eigenbasis ?
            // Something
            a = eigenbasis.col(0).transpose() * (dynamical*eigenbasis);
            // a = (eigenbasis.transpose()*dynamical)*eigenbasis.col(0);

            // std::cerr << "  calculate_a: sorting" << std::endl;
            // std::sort(a.data(),a.data()+a.size());

            // for (int i=0; i<10; ++i)
            //     std::cerr << "  a[" << i << "] = " << a[i] << std::endl;
        }
        
        void Calculate_Dynamical_Matrix_2N(const MatrixX & hessian, MatrixX & dynamical)
        {
            int nos = hessian.rows()/2;
            dynamical = MatrixX(2*nos, 2*nos);

            for (int i=0; i < nos; ++i)
            {
                for (int j=0; j < nos; ++j)
                {
                    dynamical(2*i,   2*j)   = -hessian(2*i+1, 2*j);
                    dynamical(2*i,   2*j+1) = -hessian(2*i+1, 2*j+1);
                    dynamical(2*i+1, 2*j)   =  hessian(2*i,   2*j);
                    dynamical(2*i+1, 2*j+1) =  hessian(2*i,   2*j+1);

                    // velocity(2*i, 2*j)     =   - spins[i][2]*hessian(2*i+1,2*j);
                    // velocity(2*i, 2*j+1)   =  - spins[i][2]*hessian(2*i+1,2*j+1);

                    // velocity(2*i+1, 2*j)   = spins[i][2]*hessian(2*i,2*j)     
                    // velocity(2*i+1, 2*j+1) = spins[i][2]*hessian(2*i,2*j+1)   
                }
            }
        }


        // If all needed values are known the prefactor can be calculated with this function
        void Calculate_Prefactor(int nos, int n_zero_modes_minimum, int n_zero_modes_sp, scalar volume_minimum, scalar volume_sp,
            const VectorX & _eig_min, const VectorX & _eig_sp, const VectorX & _a, scalar & prefactor, scalar & exponent)
        {
            // This implements (I think) the nu from eq. (3) from Pavel F. Bessarab - Size and Shape Dependence of Thermal Spin Transitions in Nanoislands - PRL 110, 020604 (2013)
            
            auto eig_min = 1e-3*_eig_min;
            auto eig_sp  = 1e-3*_eig_sp;
            auto a       = 1e-3*_a;

            // Calculate the exponent
            //      The exponent depends on the number of zero modes at the different states
            exponent = 0.5 * (n_zero_modes_minimum - n_zero_modes_sp);

            // QUESTION: g_e is the electron's g-factor [unitless] -> mu_s = g_e * mu_B / hbar * Spin
            scalar g_e = 2.00231930436182;

            // Calculate "me" - QUESTION: what is this?
            scalar me = 1;
            for (int i=0; i < (n_zero_modes_minimum - n_zero_modes_sp); ++i)
                me *= 2*M_PI*Utility::Constants::k_B;
            me = std::sqrt(me);

            // Calculate "m" - QUESTION: what is it?
            // TODO: the `n_zero_modes_sp+1` should be `n_zero_modes_sp+n_negative_modes_sp`
            scalar m = 1;
            if (n_zero_modes_minimum > n_zero_modes_sp+1)
            {
                for (int i = 0; i<(n_zero_modes_minimum-n_zero_modes_sp-1); ++i)
                    m /= eig_sp[i+n_zero_modes_sp+1];
            }
            else if (n_zero_modes_minimum < n_zero_modes_sp+1)
            {
                for (int i = 0; i<(n_zero_modes_sp+1-n_zero_modes_minimum); ++i)
                    m *= eig_min[i+n_zero_modes_minimum];
            }

            int j = std::max(n_zero_modes_minimum, n_zero_modes_sp+1);
            for (int i=j; i < 2*nos; ++i)
                m *= eig_min[i]/eig_sp[i];
            m = std::sqrt(m);

            // Calculate "s" - QUESTION: what is it?
            scalar s = 0;
            for (int i = n_zero_modes_sp+1; i < 2*nos; ++i)
                s += a[i]*a[i] / eig_sp[i];
            s = std::sqrt(s);

            // Calculate the prefactor
            prefactor = g_e * m * s * me * volume_sp / ( 2*M_PI * Utility::Constants::hbar * 1e-12 * volume_minimum);

            std::cerr << "exponent    = " << exponent << std::endl;
            std::cerr << "me =          " << me << std::endl;
            std::cerr << "m  =          " << m << std::endl;
            std::cerr << "s  =          " << s << std::endl;
            std::cerr << "volume_sp   = " << volume_sp << std::endl;
            std::cerr << "volume_min  = " << volume_minimum << std::endl;
            std::cerr << "hbar[meV*s] = " << Utility::Constants::hbar*1e-12 << std::endl;
            std::cerr << "prefactor   = " << prefactor << std::endl;
        }


        int find_pos(const Vector3 & vec, const vectorfield & vf)
        {
            for (int i=0; i<vf.size(); ++i)
                if (vec.isApprox(vf[i])) return i;
            return -1;
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