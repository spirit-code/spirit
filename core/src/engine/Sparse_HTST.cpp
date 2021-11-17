#ifndef SPIRIT_SKIP_HTST

#include <engine/HTST.hpp>
#include <engine/Hamiltonian_Heisenberg.hpp>
#include <engine/Manifoldmath.hpp>
#include <engine/Sparse_HTST.hpp>
#include <engine/Vectormath.hpp>
#include <utility/Constants.hpp>
#include <utility/Logging.hpp>

#include <GenEigsRealShiftSolver.h>
#include <GenEigsSolver.h> // Also includes <MatOp/DenseGenMatProd.h>
#include <MatOp/SparseSymMatProd.h>
#include <SymEigsSolver.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <fmt/format.h>
#include <fmt/ostream.h>

namespace C = Utility::Constants;

namespace Engine
{
namespace Sparse_HTST
{

void Sparse_Get_Lowest_Eigenvector( const SpMatrixX & matrix, int nos, scalar & lowest_evalue, VectorX & lowest_evec )
{
    Log( Utility::Log_Level::All, Utility::Log_Sender::HTST, "        Using Spectra to compute lowest eigenmode..." );

    VectorX evalues;
    MatrixX evectors;

    int n_steps = std::max( 2, nos );

    //  Create a Spectra solver
    Spectra::SparseSymMatProd<scalar> op( matrix );
    Spectra::SymEigsSolver<scalar, Spectra::SMALLEST_ALGE, Spectra::SparseSymMatProd<scalar>> matrix_spectrum(
        &op, 1, n_steps );

    matrix_spectrum.init();
    int nconv = matrix_spectrum.compute();

    if( matrix_spectrum.info() == Spectra::SUCCESSFUL )
    {
        evalues  = matrix_spectrum.eigenvalues().real();
        evectors = matrix_spectrum.eigenvectors().real();
    }
    else
    {
        Log( Utility::Log_Level::All, Utility::Log_Sender::HTST,
             "        Failed to calculate lowest eigenmode. Aborting!" );
        return;
    }

    lowest_evalue = evalues[0];
    lowest_evec   = evectors.col( 0 );
}

// Project vector such that it is orthogonal to all vectors in orth
void _orth_project( VectorX & vector, const std::vector<VectorX> & orth )
{
    for( const VectorX & cur : orth )
    {
        vector -= ( vector.dot( cur ) ) * cur;
    }
}

void Inverse_Shift_PowerMethod(
    int n_iter, int n_refactor, const SpMatrixX & matrix, scalar & evalue_estimate, VectorX & evec_estimate,
    std::vector<VectorX> & evecs )
{
    SpMatrixX tmp = SpMatrixX( matrix.rows(), matrix.cols() );
    tmp.setIdentity();
    tmp *= evalue_estimate;

    Eigen::SparseLU<SpMatrixX, Eigen::COLAMDOrdering<int>> solver;
    solver.analyzePattern( matrix - tmp );
    solver.factorize( matrix - tmp );

    Log( Utility::Log_Level::All, Utility::Log_Sender::HTST,
         fmt::format(
             "        ... Improve eigenpair estimate with power method for eigenvalue = {}", evalue_estimate ) );
    for( int i = 0; i < n_iter; i++ )
    {
        evec_estimate = solver.solve( evec_estimate );
        _orth_project( evec_estimate, evecs ); // Project evec_estimate orthogonally to the previous evectors
        evec_estimate.normalize();

        if( i % n_refactor == 0 )
        {
            evalue_estimate = evec_estimate.dot( matrix * evec_estimate );
            tmp.setIdentity();
            tmp *= evalue_estimate;
            solver.factorize( matrix - tmp );
            Log( Utility::Log_Level::All, Utility::Log_Sender::HTST,
                 fmt::format( "            Iteration {}/{}, e_value = {}", i, n_iter, evalue_estimate ) );
        }
    }
    evalue_estimate = evec_estimate.dot( matrix * evec_estimate );
    Log( Utility::Log_Level::All, Utility::Log_Sender::HTST,
         fmt::format( "        ... Improved eigenvalue = {}", evalue_estimate ) );
}

void Sparse_Get_Lowest_Eigenvectors_VP(
    const SpMatrixX & matrix, scalar max_evalue, scalarfield & evalues, std::vector<VectorX> & evecs )
{
    Log( Utility::Log_Level::All, Utility::Log_Sender::HTST,
         fmt::format( "    Computing eigenvalues smaller than {}", max_evalue ) );
    scalar tol           = 1e-8;
    int n_log_step       = 20000;
    int max_iter         = 10 * n_log_step;
    int n_iter_power     = 250;
    int n_power_refactor = 50;
    scalar cur           = 2 * tol;
    scalar m             = 0.01;
    scalar step_size     = 1e-4;
    int n_iter           = 0;
    int nos              = matrix.rows() / 2;

    scalar sigma_shift = std::max( scalar( 5.0 ), 2 * scalar( max_evalue ) );

    VectorX gradient      = VectorX::Zero( 2 * nos );
    VectorX gradient_prev = VectorX::Zero( 2 * nos );
    VectorX velocity      = VectorX::Zero( 2 * nos );
    scalar cur_evalue_estimate;
    scalar fnorm2, ratio, proj;

    scalar max_grad_comp = 0;
    bool run             = true;

    // We try to find the lowest n_values eigenvalue/vector pairs
    while( run )
    {
        VectorX x = VectorX::Random( 2 * nos ); // Initialize solver with random normalized vector
        x.normalize();

        fnorm2 = 2 * tol * tol;
        n_iter = 0;
        gradient_prev.setZero();
        velocity.setZero();

        bool search = true;
        Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, fmt::format( "        Search for eigenpair" ) );

        while( search )
        {
            // Compute gradient of unnormalized Rayleigh quotient
            gradient = 2 * matrix * x;

            cur_evalue_estimate = 0.5 * x.dot( gradient ); // Update the current estimate of our evalue
            for( int i = 0; i < evecs.size(); i++ )
            {
                gradient
                    += 2 * ( sigma_shift - evalues[i] ) * ( evecs[i].dot( x ) )
                       * evecs
                           [i]; // Add the shift so that we dont land on the same eigenvalues we had before. Effectively
                                // H -> H + (sigma - lambda) * v^T v, where (lambda, v) is an eigenvalue, eigenvector pair
            }

            // Project the gradient orthogonally wrt to x and the previous eigenvectors
            _orth_project( gradient, { x } );

            velocity = 0.5 * ( gradient + gradient_prev ) / m;
            fnorm2   = gradient.squaredNorm();

            proj  = velocity.dot( gradient );
            ratio = proj / fnorm2;

            if( proj <= 0 )
                velocity.setZero();
            else
                velocity = gradient * ratio;

            // Update x
            x -= step_size * velocity + 0.5 / m * step_size * gradient;

            // Re-orthogonalize
            for( int i = 0; i < evecs.size(); i++ )
            {
                x -= ( x.dot( evecs[i] ) ) * x;
            }
            // Re-normalize
            x.normalize();

            // Update prev gradient
            gradient_prev = gradient;

            // Increment n_iter
            n_iter++;

            max_grad_comp = 0;
            for( const auto & g : gradient )
                max_grad_comp = std::max( std::abs( g ), max_grad_comp );

            if( n_iter % n_log_step == 0 )
                Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST,
                     fmt::format(
                         "        ... Iteration {} (max. {} [{}%]): Evalue estimate = {}, Grad. norm = {} (> {})",
                         n_iter, max_iter, n_iter / max_iter * 100, cur_evalue_estimate, max_grad_comp, tol ) );

            search = max_grad_comp > tol && n_iter < max_iter;
        }

        // We improve the eigenpair estimate with the inverse shift power method (This may be necessary for accurate
        // cancellation of zero-modes)
        Inverse_Shift_PowerMethod( n_iter_power, n_power_refactor, matrix, cur_evalue_estimate, x, evecs );

        // Ideally we have found one eigenvalue/vector pair now
        // We save the eigenvalue
        evecs.push_back( x );
        evalues.push_back( x.dot( matrix * x ) );

        Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST,
             fmt::format( "        ... Found an eigenpair after {} iterations", n_iter ) );
        Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST,
             fmt::format( "        ... Eigenvalue  = {}", evalues.back() ) );
        if( 2 * nos >= 4 )
            Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST,
                 fmt::format(
                     "        ... Eigenvector = ({}, {}, {}, ..., {})", evecs.back()[0], evecs.back()[1],
                     evecs.back()[2], evecs.back()[2 * nos - 1] ) );
        if( evalues.back() > max_evalue )
        {
            Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST,
                 fmt::format( "        No more eigenvalues < {} found. Stopping.", max_evalue ) );
            run = false;
        }
    }
}

// Note the two images should correspond to one minimum and one saddle point
// Non-extremal images may yield incorrect Hessians and thus incorrect results
void Calculate( Data::HTST_Info & htst_info )
{
    Log( Utility::Log_Level::All, Utility::Log_Sender::HTST, "Sparse Prefactor calculation" );
    bool lowest_mode_spectra    = false;
    htst_info.sparse            = true;
    htst_info.n_eigenmodes_keep = 0;

    const scalar epsilon       = 1e-4;
    const scalar epsilon_force = 1e-8;

    auto & image_minimum = *htst_info.minimum->spins;
    auto & image_sp      = *htst_info.saddle_point->spins;

    int nos = image_minimum.size();

    Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Saving NO eigenvectors." );

    vectorfield force_tmp( nos, { 0, 0, 0 } );
    std::vector<std::string> block;

    // TODO
    bool is_afm = false;

    // The gradient (unprojected)
    Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST,
         "    Evaluation of the gradient at the initial configuration..." );
    vectorfield gradient_minimum( nos, { 0, 0, 0 } );
    htst_info.minimum->hamiltonian->Gradient( image_minimum, gradient_minimum );

    // Check if the configuration is actually an extremum
    Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST,
         "    Checking if initial configuration is an extremum..." );
    Vectormath::set_c_a( 1, gradient_minimum, force_tmp );
    Manifoldmath::project_tangential( force_tmp, image_minimum );
    scalar fmax_minimum = Vectormath::max_norm( force_tmp );
    if( fmax_minimum > epsilon_force )
    {
        Log( Utility::Log_Level::Error, Utility::Log_Sender::All,
             fmt::format(
                 "HTST: the initial configuration is not a converged minimum, its max. torque is above the threshold "
                 "({} > {})!",
                 fmax_minimum, epsilon_force ) );
        return;
    }

    // The gradient (unprojected)
    Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST,
         "    Evaluation of the gradient at the transition configuration..." );
    vectorfield gradient_sp( nos, { 0, 0, 0 } );
    htst_info.saddle_point->hamiltonian->Gradient( image_sp, gradient_sp );

    // Check if the configuration is actually an extremum
    Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST,
         "    Checking if transition configuration is an extremum..." );
    Vectormath::set_c_a( 1, gradient_sp, force_tmp );
    Manifoldmath::project_tangential( force_tmp, image_sp );
    scalar fmax_sp = Vectormath::max_norm( force_tmp );
    if( fmax_sp > epsilon_force )
    {
        Log( Utility::Log_Level::Error, Utility::Log_Sender::All,
             fmt::format(
                 "HTST: the transition configuration is not a converged saddle point, its max. torque is above the "
                 "threshold ({} > {})!",
                 fmax_sp, epsilon_force ) );
        return;
    }

    ////////////////////////////////////////////////////////////////////////
    // Saddle point
    int n_zero_modes_sp    = 0;
    scalarfield evalues_sp = scalarfield( 0 );
    {
        Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "Calculation for the Saddle Point" );

        Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Evaluate tangent basis ..." );
        SpMatrixX tangent_basis = SpMatrixX( 3 * nos, 2 * nos );
        Manifoldmath::sparse_tangent_basis_spherical( image_sp, tangent_basis );

        // Evaluation of the Hessian...
        Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Evaluate the Hessian..." );
        SpMatrixX sparse_hessian_sp( 3 * nos, 3 * nos );
        htst_info.saddle_point->hamiltonian->Sparse_Hessian( image_sp, sparse_hessian_sp );

        // Transform into geodesic Hessian
        Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Transform Hessian into geodesic Hessian..." );
        SpMatrixX sparse_hessian_sp_geodesic_3N( 3 * nos, 3 * nos );
        sparse_hessian_bordered_3N( image_sp, gradient_sp, sparse_hessian_sp, sparse_hessian_sp_geodesic_3N );
        SpMatrixX sparse_hessian_sp_geodesic_2N
            = tangent_basis.transpose() * sparse_hessian_sp_geodesic_3N * tangent_basis;

        Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST,
             "    Sparse LU Decomposition of geodesic Hessian..." );
        Eigen::SparseLU<SpMatrixX, Eigen::COLAMDOrdering<int>> solver;
        solver.analyzePattern( sparse_hessian_sp_geodesic_2N );
        solver.factorize( sparse_hessian_sp_geodesic_2N );

        Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Evaluate lowest eigenmode of the Hessian..." );

        std::vector<VectorX> evecs_sp = std::vector<VectorX>( 0 );
        Sparse_Get_Lowest_Eigenvectors_VP( sparse_hessian_sp_geodesic_2N, epsilon, evalues_sp, evecs_sp );
        scalar lowest_evalue     = evalues_sp[0];
        VectorX & lowest_evector = evecs_sp[0];

        htst_info.det_sp = solver.logAbsDeterminant() - std::log( -lowest_evalue );

        // Check if lowest eigenvalue < 0 (else it's not a SP)
        Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Check if actually a saddle point..." );
        if( lowest_evalue > -epsilon )
        {
            Log( Utility::Log_Level::Error, Utility::Log_Sender::All,
                 fmt::format(
                     "HTST: the transition configuration is not a saddle point, its lowest eigenvalue is above the "
                     "threshold ({} > {})!",
                     lowest_evalue, -epsilon ) );
            return;
        }
        // Check if second-lowest eigenvalue < 0 (higher-order SP)
        Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Check if higher order saddle point..." );
        int n_negative = 0;
        for( int i = 0; i < evalues_sp.size(); ++i )
            if( evalues_sp[i] < -epsilon )
                ++n_negative;

        if( n_negative > 1 )
        {
            Log( Utility::Log_Level::Error, Utility::Log_Sender::All,
                 fmt::format( "HTST: the image you passed is a higher order saddle point (N={})!", n_negative ) );
            return;
        }

        // Perpendicular velocity
        Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Calculate dynamical contribution" );

        Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Evaluate the dynamical matrix" );
        SpMatrixX velocity( 3 * nos, 3 * nos );
        Sparse_Calculate_Dynamical_Matrix(
            image_sp, htst_info.saddle_point->geometry->mu_s, sparse_hessian_sp_geodesic_3N, velocity );
        SpMatrixX projected_velocity = tangent_basis.transpose() * velocity * tangent_basis;

        Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Solving H^-1 V q_1 ..." );
        VectorX x( 2 * nos );
        x           = solver.solve( projected_velocity.transpose() * lowest_evector );
        htst_info.s = std::sqrt( lowest_evector.transpose() * projected_velocity * x );
        // Checking for zero modes at the saddle point...
        Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST,
             "    Checking for zero modes at the saddle point..." );
        for( int i = 0; i < evalues_sp.size(); ++i )
        {
            if( std::abs( evalues_sp[i] ) <= epsilon )
                ++n_zero_modes_sp;
        }
        // Deal with zero modes if any (calculate volume)
        htst_info.volume_sp = 1;
        if( n_zero_modes_sp > 0 )
        {
            Log( Utility::Log_Level::All, Utility::Log_Sender::HTST,
                 fmt::format( "ZERO MODES AT SADDLE POINT (N={})", n_zero_modes_sp ) );
            htst_info.volume_sp = HTST::Calculate_Zero_Volume( htst_info.saddle_point );
        }
    }

    // TODO  // End saddle point
    ////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////
    // Initial state minimum
    int n_zero_modes_minimum = 0;
    scalarfield evalues_min  = scalarfield( 0 );
    {
        Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "Calculation for the Minimum" );

        Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Evaluate tangent basis ..." );
        SpMatrixX tangent_basis = SpMatrixX( 3 * nos, 2 * nos );
        Manifoldmath::sparse_tangent_basis_spherical( image_minimum, tangent_basis );

        // Evaluation of the Hessian...
        Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Evaluate the Hessian..." );
        SpMatrixX sparse_hessian_minimum = SpMatrixX( 3 * nos, 3 * nos );
        htst_info.minimum->hamiltonian->Sparse_Hessian( image_minimum, sparse_hessian_minimum );

        // Transform into geodesic Hessian
        Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Transforming Hessian into geodesic Hessian..." );
        SpMatrixX sparse_hessian_geodesic_min_3N = SpMatrixX( 3 * nos, 3 * nos );
        sparse_hessian_bordered_3N(
            image_minimum, gradient_minimum, sparse_hessian_minimum, sparse_hessian_geodesic_min_3N );
        SpMatrixX sparse_hessian_geodesic_min_2N
            = tangent_basis.transpose() * sparse_hessian_geodesic_min_3N * tangent_basis;

        Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST,
             "    Sparse LU Decomposition of geodesic Hessian..." );
        Eigen::SparseLU<SpMatrixX, Eigen::COLAMDOrdering<int>> solver;
        solver.analyzePattern( sparse_hessian_geodesic_min_2N );
        solver.factorize( sparse_hessian_geodesic_min_2N );
        htst_info.det_min = solver.logAbsDeterminant();

        // Calculate modes at minimum (needed for zero-mode volume)
        std::vector<VectorX> evecs_min = std::vector<VectorX>( 0 );
        Sparse_Get_Lowest_Eigenvectors_VP( sparse_hessian_geodesic_min_2N, epsilon, evalues_min, evecs_min );

        // Checking for zero modes at the minimum..
        Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Checking for zero modes at the minimum ..." );
        for( int i = 0; i < evalues_min.size(); ++i )
        {
            if( std::abs( evalues_min[i] ) <= epsilon )
                ++n_zero_modes_minimum;
            if( evalues_min[i] < 0 )
            {
                Log(
                    Utility::Log_Level::Warning, Utility::Log_Sender::HTST,
                    fmt::format(
                        "    Minimum has a negative mode with eigenvalue = {}!",
                        evalues_min[i] ) ); // The Question is if we should terminate the calculation here or allow to
                                            // continue since often the negatives cancel sqrt(-x) * sqrt(-x) = sqrt(x^2)
            }
        }
        // Deal with zero modes if any (calculate volume)
        htst_info.volume_min = 1;
        if( n_zero_modes_minimum > 0 )
        {
            Log( Utility::Log_Level::All, Utility::Log_Sender::HTST,
                 fmt::format( "ZERO MODES AT MINIMUM (N={})", n_zero_modes_minimum ) );
            htst_info.volume_min = HTST::Calculate_Zero_Volume( htst_info.minimum );
        }
    }
    // End initial state minimum
    ////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////
    // Calculation of the prefactor...
    Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "Calculating prefactor..." );

    // Calculate the exponent for the temperature-dependence of the prefactor
    //      The exponent depends on the number of zero modes at the different states
    htst_info.temperature_exponent = 0.5 * ( n_zero_modes_minimum - n_zero_modes_sp );

    // Calculate "me"
    htst_info.me = std::pow( 2 * C::Pi * C::k_B, htst_info.temperature_exponent );

    // Calculate Omega_0, i.e. the entropy contribution
    htst_info.Omega_0 = std::sqrt( std::exp( htst_info.det_min - htst_info.det_sp ) );

    scalar zero_mode_factor = 1;
    for( int i = 0; i < n_zero_modes_minimum; i++ )
        zero_mode_factor /= std::abs( evalues_min[i] ); // We can take the abs here and in the determinants, because in
                                                        // the end we know the result must be positive

    for( int i = 0; i < n_zero_modes_sp; i++ )
        zero_mode_factor *= std::abs( evalues_sp[i + 1] ); // We can take the abs here and in the determinants, because
                                                           // in the end we know the result must be positive

    zero_mode_factor = std::sqrt( zero_mode_factor );

    htst_info.Omega_0 *= zero_mode_factor;

    // Calculate the prefactor
    htst_info.prefactor_dynamical = htst_info.me * htst_info.volume_sp / htst_info.volume_min * htst_info.s;
    htst_info.prefactor
        = C::g_e / ( C::hbar * 1e-12 ) * htst_info.Omega_0 * htst_info.prefactor_dynamical / ( 2 * C::Pi );

    Log.SendBlock(
        Utility::Log_Level::All, Utility::Log_Sender::HTST,
        { "---- Prefactor calculation successful!",
          fmt::format( "exponent      = {:^20e}", htst_info.temperature_exponent ),
          fmt::format( "me            = {:^20e}", htst_info.me ),
          fmt::format( "m = Omega_0   = {:^20e}", htst_info.Omega_0 ),
          fmt::format( "s             = {:^20e}", htst_info.s ),
          fmt::format( "volume_sp     = {:^20e}", htst_info.volume_sp ),
          fmt::format( "volume_min    = {:^20e}", htst_info.volume_min ),
          fmt::format( "log |det_min| = {:^20e}", htst_info.det_min ),
          fmt::format( "log |det_sp|  = {:^20e}", htst_info.det_sp ),
          fmt::format( "0-mode factor = {:^20e}", zero_mode_factor ),
          fmt::format( "hbar[meV*s]   = {:^20e}", C::hbar * 1e-12 ),
          fmt::format( "v = dynamical prefactor = {:^20e}", htst_info.prefactor_dynamical ),
          fmt::format( "prefactor               = {:^20e}", htst_info.prefactor ) },
        -1, -1 );
}

void Sparse_Calculate_Dynamical_Matrix(
    const vectorfield & spins, const scalarfield & mu_s, const SpMatrixX & hessian, SpMatrixX & velocity )
{
    constexpr scalar epsilon = 1e-10;
    int nos                  = spins.size();

    typedef Eigen::Triplet<scalar> T;
    std::vector<T> tripletList;
    tripletList.reserve( hessian.nonZeros() );

    auto levi_civita = []( int i, int j, int k ) { return -0.5 * ( j - i ) * ( k - j ) * ( i - k ); };

    // We first compute the effective field temporary
    auto b_eff = vectorfield( nos, { 0, 0, 0 } );
    for( int k = 0; k < hessian.outerSize(); ++k )
    {
        for( SpMatrixX::InnerIterator it( hessian, k ); it; ++it )
        {
            int row = it.row(), col = it.col();
            scalar h = it.value();

            for( int nu = 0; nu < 3; nu++ )
            {
                for( int gamma = 0; gamma < 3; gamma++ )
                {
                    if( ( row - nu ) % 3 != 0 || ( col - gamma ) % 3 != 0 || nu > row || gamma > col )
                        continue;
                    int i = ( row - nu ) / 3.0;
                    int j = ( col - gamma ) / 3.0;
                    b_eff[i][nu] += h * spins[j][gamma] / mu_s[i];
                }
            }
        }
    }

    // Add the contributions from the effective field
    for( int i = 0; i < nos; i++ )
    {
        for( int alpha = 0; alpha < 3; alpha++ )
        {
            for( int beta = 0; beta < 3; beta++ )
            {
                for( int nu = 0; nu < 3; nu++ )
                {
                    scalar res = levi_civita( alpha, beta, nu ) * b_eff[i][nu];
                    if( std::abs( res ) > epsilon )
                        tripletList.push_back( T( 3 * i + alpha, 3 * i + beta, res ) );
                }
            }
        }
    }

    // Iterate over non zero entries of hessian
    for( int k = 0; k < hessian.outerSize(); ++k )
    {
        for( SpMatrixX::InnerIterator it( hessian, k ); it; ++it )
        {
            int row = it.row(), col = it.col();
            scalar h = it.value();

            for( int mu = 0; mu < 3; mu++ )
            {
                for( int nu = 0; nu < 3; nu++ )
                {
                    for( int alpha = 0; alpha < 3; alpha++ )
                    {
                        for( int beta = 0; beta < 3; beta++ )
                        {
                            if( ( row - nu ) % 3 != 0 || ( col - beta ) % 3 != 0 || nu > row || beta > col )
                                continue;

                            int i      = ( row - nu ) / 3.0;
                            int j      = ( col - beta ) / 3.0;
                            scalar res = levi_civita( alpha, mu, nu ) * spins[i][mu] * h / mu_s[i];
                            tripletList.push_back( T( 3 * i + alpha, 3 * j + beta, res ) );
                        }
                    }
                }
            }
        }
    }

    velocity.setFromTriplets( tripletList.begin(), tripletList.end() );
}

void sparse_hessian_bordered_3N(
    const vectorfield & image, const vectorfield & gradient, const SpMatrixX & hessian, SpMatrixX & hessian_out )
{
    // Calculates a 3Nx3N matrix in the bordered Hessian approach and transforms it into the tangent basis,
    // making the result a 2Nx2N matrix. The bordered Hessian's Lagrange multipliers assume a local extremum.

    int nos = image.size();
    VectorX lambda( nos );
    for( int i = 0; i < nos; ++i )
        lambda[i] = image[i].normalized().dot( gradient[i] );

    // Construct hessian_out
    typedef Eigen::Triplet<scalar> T;
    std::vector<T> tripletList;
    tripletList.reserve( hessian.nonZeros() + 3 * nos );

    // Iterate over non zero entries of hesiian
    for( int k = 0; k < hessian.outerSize(); ++k )
    {
        for( SpMatrixX::InnerIterator it( hessian, k ); it; ++it )
        {
            tripletList.push_back( T( it.row(), it.col(), it.value() ) );
        }
        int j = k % 3;
        int i = ( k - j ) / 3;
        tripletList.push_back( T( k, k, -lambda[i] ) ); // Correction to the diagonal
    }
    hessian_out.setFromTriplets( tripletList.begin(), tripletList.end() );
}

// NOTE WE ASSUME A SELFADJOINT MATRIX
void Sparse_Eigen_Decomposition( const SpMatrixX & matrix, VectorX & evalues, MatrixX & evectors )
{
    // Create a Spectra solver
    Eigen::SelfAdjointEigenSolver<SpMatrixX> matrix_solver( matrix );
    evalues  = matrix_solver.eigenvalues().real();
    evectors = matrix_solver.eigenvectors().real();
}

void Sparse_Geodesic_Eigen_Decomposition(
    const vectorfield & image, const vectorfield & gradient, const SpMatrixX & hessian, SpMatrixX & hessian_geodesic_3N,
    SpMatrixX & hessian_geodesic_2N, SpMatrixX & tangent_basis, VectorX & eigenvalues, MatrixX & eigenvectors )
{
    Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "---------- Sparse Geodesic Eigen Decomposition" );

    int nos = image.size();

    // Calculate geodesic Hessian in 3N-representation
    hessian_geodesic_3N = SpMatrixX( 3 * nos, 3 * nos );
    sparse_hessian_bordered_3N( image, gradient, hessian, hessian_geodesic_3N );

    // Transform into geodesic Hessian
    Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Transforming Hessian into geodesic Hessian..." );
    hessian_geodesic_2N = SpMatrixX( 2 * nos, 2 * nos );
    hessian_geodesic_2N = tangent_basis.transpose() * hessian_geodesic_3N * tangent_basis;

    // Calculate full eigenspectrum
    Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Calculation of full eigenspectrum..." );

    eigenvalues  = VectorX::Zero( 2 * nos );
    eigenvectors = MatrixX::Zero( 2 * nos, 2 * nos );

    Sparse_Eigen_Decomposition( hessian_geodesic_2N, eigenvalues, eigenvectors );

    Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "---------- Sparse Geodesic Eigen Decomposition Done" );
}

} // namespace Sparse_HTST
} // namespace Engine

#endif