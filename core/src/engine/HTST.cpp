#ifndef SPIRIT_SKIP_HTST

#include <engine/HTST.hpp>
#include <engine/Hamiltonian_Heisenberg.hpp>
#include <engine/Manifoldmath.hpp>
#include <engine/Vectormath.hpp>
#include <utility/Constants.hpp>
#include <utility/Logging.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
// #include <Eigen/Array>
#include <Eigen/Eigenvalues>
//#include <unsupported/Eigen/CXX11/Tensor>
#include <GenEigsRealShiftSolver.h>
#include <GenEigsSolver.h> // Also includes <MatOp/DenseGenMatProd.h>

#include <fmt/format.h>
#include <fmt/ostream.h>

namespace C = Utility::Constants;

namespace Engine
{
namespace HTST
{

// Note the two images should correspond to one minimum and one saddle point
// Non-extremal images may yield incorrect Hessians and thus incorrect results
void Calculate( Data::HTST_Info & htst_info, int n_eigenmodes_keep )
{
    Log( Utility::Log_Level::All, Utility::Log_Sender::HTST, "---- Prefactor calculation" );
    htst_info.sparse           = false;
    const scalar epsilon       = 1e-4;
    const scalar epsilon_force = 1e-8;

    auto & image_minimum = *htst_info.minimum->spins;
    auto & image_sp      = *htst_info.saddle_point->spins;

    int nos = image_minimum.size();

    if( n_eigenmodes_keep < 0 )
        n_eigenmodes_keep = 2 * nos;
    n_eigenmodes_keep           = std::min( 2 * nos, n_eigenmodes_keep );
    htst_info.n_eigenmodes_keep = n_eigenmodes_keep;
    Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST,
         fmt::format( "    Saving the first {} eigenvectors.", n_eigenmodes_keep ) );

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
    {
        Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "Calculation for the Saddle Point" );

        // Evaluation of the Hessian...
        MatrixX hessian_sp = MatrixX::Zero( 3 * nos, 3 * nos );
        Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Evaluation of the Hessian..." );
        htst_info.saddle_point->hamiltonian->Hessian( image_sp, hessian_sp );

        // Eigendecomposition
        Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Eigendecomposition..." );
        MatrixX hessian_geodesic_sp_3N = MatrixX::Zero( 3 * nos, 3 * nos );
        MatrixX hessian_geodesic_sp_2N = MatrixX::Zero( 2 * nos, 2 * nos );
        htst_info.eigenvalues_sp       = VectorX::Zero( 2 * nos );
        htst_info.eigenvectors_sp      = MatrixX::Zero( 2 * nos, 2 * nos );
        Geodesic_Eigen_Decomposition(
            image_sp, gradient_sp, hessian_sp, hessian_geodesic_sp_3N, hessian_geodesic_sp_2N, htst_info.eigenvalues_sp,
            htst_info.eigenvectors_sp );

        // Print some eigenvalues
        block = std::vector<std::string>{ "10 lowest eigenvalues at saddle point:" };
        for( int i = 0; i < 10; ++i )
            block.push_back( fmt::format(
                "ew[{}]={:^20e}   ew[{}]={:^20e}", i, htst_info.eigenvalues_sp[i], i + 2 * nos - 10,
                htst_info.eigenvalues_sp[i + 2 * nos - 10] ) );
        Log.SendBlock( Utility::Log_Level::Info, Utility::Log_Sender::HTST, block, -1, -1 );

        // Check if lowest eigenvalue < 0 (else it's not a SP)
        Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "Checking if actually a saddle point..." );
        if( htst_info.eigenvalues_sp[0] > -epsilon )
        {
            Log( Utility::Log_Level::Error, Utility::Log_Sender::All,
                 fmt::format(
                     "HTST: the transition configuration is not a saddle point, its lowest eigenvalue is above the "
                     "threshold ({} > {})!",
                     htst_info.eigenvalues_sp[0], -epsilon ) );
            return;
        }

        // Check if second-lowest eigenvalue < 0 (higher-order SP)
        Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "Checking if higher order saddle point..." );
        int n_negative = 0;
        for( int i = 0; i < htst_info.eigenvalues_sp.size(); ++i )
        {
            if( htst_info.eigenvalues_sp[i] < -epsilon )
                ++n_negative;
        }
        if( n_negative > 1 )
        {
            Log( Utility::Log_Level::Error, Utility::Log_Sender::All,
                 fmt::format( "HTST: the image you passed is a higher order saddle point (N={})!", n_negative ) );
            return;
        }

        // Perpendicular velocity
        Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST,
             "Calculating perpendicular velocity at saddle point ('a' factors)..." );
        // Calculation of the 'a' parameters...
        htst_info.perpendicular_velocity = VectorX::Zero( 2 * nos );
        MatrixX basis_sp                 = MatrixX::Zero( 3 * nos, 2 * nos );
        Manifoldmath::tangent_basis_spherical( image_sp, basis_sp );
        // Manifoldmath::tangent_basis(image_sp, basis_sp);
        // Calculate_Perpendicular_Velocity_2N(image_sp, hessian_geodesic_sp_2N, basis_sp, htst_info.eigenvectors_sp,
        // perpendicular_velocity_sp);
        Calculate_Perpendicular_Velocity(
            image_sp, htst_info.saddle_point->geometry->mu_s, hessian_geodesic_sp_3N, basis_sp,
            htst_info.eigenvectors_sp, htst_info.perpendicular_velocity );

        // Reduce the number of saved eigenmodes
        htst_info.eigenvalues_sp.conservativeResize( 2 * nos );
        htst_info.eigenvectors_sp.conservativeResize( 2 * nos, n_eigenmodes_keep );
    }
    // End saddle point
    ////////////////////////////////////////////////////////////////////////

    // Checking for zero modes at the saddle point...
    Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "Checking for zero modes at the saddle point..." );
    int n_zero_modes_sp = 0;
    for( int i = 0; i < htst_info.eigenvalues_sp.size(); ++i )
    {
        if( std::abs( htst_info.eigenvalues_sp[i] ) <= epsilon )
            ++n_zero_modes_sp;
    }

    // Deal with zero modes if any (calculate volume)
    htst_info.volume_sp = 1;
    if( n_zero_modes_sp > 0 )
    {
        Log( Utility::Log_Level::All, Utility::Log_Sender::HTST,
             fmt::format( "ZERO MODES AT SADDLE POINT (N={})", n_zero_modes_sp ) );

        if( is_afm )
            htst_info.volume_sp = Calculate_Zero_Volume( htst_info.saddle_point );
        else
            htst_info.volume_sp = Calculate_Zero_Volume( htst_info.saddle_point );
    }

    // Calculate "s"
    htst_info.s = 0;
    for( int i = n_zero_modes_sp + 1; i < 2 * nos; ++i )
        htst_info.s += std::pow( htst_info.perpendicular_velocity[i], 2 ) / htst_info.eigenvalues_sp[i];
    htst_info.s = std::sqrt( htst_info.s );

    ////////////////////////////////////////////////////////////////////////
    // Initial state minimum
    {
        Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "Calculation for the Minimum" );

        // Evaluation of the Hessian...
        MatrixX hessian_minimum = MatrixX::Zero( 3 * nos, 3 * nos );
        Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Evaluation of the Hessian..." );
        htst_info.minimum->hamiltonian->Hessian( image_minimum, hessian_minimum );

        // Eigendecomposition
        Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Eigendecomposition..." );
        MatrixX hessian_geodesic_minimum_3N = MatrixX::Zero( 3 * nos, 3 * nos );
        MatrixX hessian_geodesic_minimum_2N = MatrixX::Zero( 2 * nos, 2 * nos );
        htst_info.eigenvalues_min           = VectorX::Zero( 2 * nos );
        htst_info.eigenvectors_min          = MatrixX::Zero( 2 * nos, 2 * nos );
        Geodesic_Eigen_Decomposition(
            image_minimum, gradient_minimum, hessian_minimum, hessian_geodesic_minimum_3N, hessian_geodesic_minimum_2N,
            htst_info.eigenvalues_min, htst_info.eigenvectors_min );

        // Print some eigenvalues
        block = std::vector<std::string>{ "10 lowest eigenvalues at minimum:" };
        for( int i = 0; i < 10; ++i )
            block.push_back( fmt::format(
                "ew[{}]={:^20e}   ew[{}]={:^20e}", i, htst_info.eigenvalues_min[i], i + 2 * nos - 10,
                htst_info.eigenvalues_min[i + 2 * nos - 10] ) );
        Log.SendBlock( Utility::Log_Level::Info, Utility::Log_Sender::HTST, block, -1, -1 );

        // Check for eigenvalues < 0 (i.e. not a minimum)
        Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "Checking if actually a minimum..." );
        if( htst_info.eigenvalues_min[0] < -epsilon )
        {
            Log( Utility::Log_Level::Error, Utility::Log_Sender::All,
                 fmt::format(
                     "HTST: the initial configuration is not a minimum, its lowest eigenvalue is below the threshold "
                     "({} < {})!",
                     htst_info.eigenvalues_min[0], -epsilon ) );
            return;
        }

        // Reduce the number of saved eigenmodes
        htst_info.eigenvalues_min.conservativeResize( 2 * nos );
        htst_info.eigenvectors_min.conservativeResize( 2 * nos, n_eigenmodes_keep );
    }
    // End initial state minimum
    ////////////////////////////////////////////////////////////////////////

    // Checking for zero modes at the minimum...
    Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "Checking for zero modes at the minimum..." );
    int n_zero_modes_minimum = 0;
    for( int i = 0; i < htst_info.eigenvalues_min.size(); ++i )
    {
        if( std::abs( htst_info.eigenvalues_min[i] ) <= epsilon )
            ++n_zero_modes_minimum;
    }

    // Deal with zero modes if any (calculate volume)
    htst_info.volume_min = 1;
    if( n_zero_modes_minimum > 0 )
    {
        Log( Utility::Log_Level::All, Utility::Log_Sender::HTST,
             fmt::format( "ZERO MODES AT MINIMUM (N={})", n_zero_modes_minimum ) );

        if( is_afm )
            htst_info.volume_min = Calculate_Zero_Volume( htst_info.minimum );
        else
            htst_info.volume_min = Calculate_Zero_Volume( htst_info.minimum );
    }

    ////////////////////////////////////////////////////////////////////////
    // Calculation of the prefactor...
    Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "Calculating prefactor..." );

    // Calculate the exponent for the temperature-dependence of the prefactor
    //      The exponent depends on the number of zero modes at the different states
    htst_info.temperature_exponent = 0.5 * ( n_zero_modes_minimum - n_zero_modes_sp );

    // Calculate "me"
    htst_info.me = std::pow( 2 * C::Pi * C::k_B, htst_info.temperature_exponent );

    // Calculate Omega_0, i.e. the entropy contribution
    htst_info.Omega_0 = 1;
    if( n_zero_modes_minimum > n_zero_modes_sp + 1 )
    {
        for( int i = n_zero_modes_sp + 1; i < n_zero_modes_minimum; ++i )
            htst_info.Omega_0 /= std::sqrt( htst_info.eigenvalues_sp[i] );
    }
    else if( n_zero_modes_minimum < n_zero_modes_sp + 1 )
    {
        for( int i = n_zero_modes_minimum; i < ( n_zero_modes_sp + 1 ); ++i )
            htst_info.Omega_0 *= std::sqrt( htst_info.eigenvalues_min[i] );
    }
    for( int i = std::max( n_zero_modes_minimum, n_zero_modes_sp + 1 ); i < 2 * nos; ++i )
        htst_info.Omega_0 *= std::sqrt( htst_info.eigenvalues_min[i] / htst_info.eigenvalues_sp[i] );

    // Calculate the prefactor
    htst_info.prefactor_dynamical = htst_info.me * htst_info.volume_sp / htst_info.volume_min * htst_info.s;
    htst_info.prefactor
        = C::g_e / ( C::hbar * 1e-12 ) * htst_info.Omega_0 * htst_info.prefactor_dynamical / ( 2 * C::Pi );

    Log.SendBlock(
        Utility::Log_Level::All, Utility::Log_Sender::HTST,
        { "---- Prefactor calculation successful!",
          fmt::format( "exponent    = {:^20e}", htst_info.temperature_exponent ),
          fmt::format( "me          = {:^20e}", htst_info.me ),
          fmt::format( "m = Omega_0 = {:^20e}", htst_info.Omega_0 ),
          fmt::format( "s           = {:^20e}", htst_info.s ),
          fmt::format( "volume_sp   = {:^20e}", htst_info.volume_sp ),
          fmt::format( "volume_min  = {:^20e}", htst_info.volume_min ),
          fmt::format( "hbar[meV*s] = {:^20e}", C::hbar * 1e-12 ),
          fmt::format( "v = dynamical prefactor = {:^20e}", htst_info.prefactor_dynamical ),
          fmt::format( "prefactor               = {:^20e}", htst_info.prefactor ) },
        -1, -1 );
}

scalar Calculate_Zero_Volume( const std::shared_ptr<Data::Spin_System> system )
{
    int nos                = system->geometry->nos;
    auto & n_cells         = system->geometry->n_cells;
    auto & spins           = *system->spins;
    auto & spin_positions  = system->geometry->positions;
    auto & geometry        = *system->geometry;
    auto & bravais_vectors = system->geometry->bravais_vectors;

    // Dimensionality of the zero mode
    int zero_mode_dimensionality = 0;
    Vector3 zero_mode_length{ 0, 0, 0 };
    for( int ibasis = 0; ibasis < 3; ++ibasis )
    {
        // Only a periodical direction can be a true zero mode
        if( system->hamiltonian->boundary_conditions[ibasis] && geometry.n_cells[ibasis] > 1 )
        {
            // Vector3 shift_pos, test_pos;
            vectorfield spins_shifted( nos, Vector3{ 0, 0, 0 } );

            int shift = 0;
            if( ibasis == 0 )
                shift = geometry.n_cell_atoms;
            else if( ibasis == 1 )
                shift = geometry.n_cell_atoms * geometry.n_cells[0];
            else if( ibasis == 2 )
                shift = geometry.n_cell_atoms * geometry.n_cells[0] * geometry.n_cells[1];

            for( int isite = 0; isite < nos; ++isite )
            {
                spins_shifted[( isite + shift ) % nos] = spins[isite];
            }

            zero_mode_length[ibasis] = Manifoldmath::dist_geodesic( spins, spins_shifted );

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
        scalar area_factor = ( bravais_vectors[0].normalized().cross( bravais_vectors[1].normalized() ) ).norm();
        zero_volume        = zero_mode_length[0] * zero_mode_length[1] * area_factor;
    }
    else if( zero_mode_dimensionality == 3 )
    {
        scalar volume_factor = std::abs( ( bravais_vectors[0].normalized().cross( bravais_vectors[1].normalized() ) )
                                             .dot( bravais_vectors[2].normalized() ) );
        zero_volume          = zero_mode_length[0] * zero_mode_length[1] * zero_mode_length[2] * volume_factor;
    }

    Log.SendBlock(
        Utility::Log_Level::Info, Utility::Log_Sender::HTST,
        { fmt::format( "ZV zero mode dimensionality = {}", zero_mode_dimensionality ),
          fmt::format( "ZV         zero mode length = {}", zero_mode_length.transpose() ),
          fmt::format( "ZV = {}", zero_volume ) },
        -1, -1 );

    // Return
    return zero_volume;
}

void Calculate_Perpendicular_Velocity(
    const vectorfield & spins, const scalarfield & mu_s, const MatrixX & hessian, const MatrixX & basis,
    const MatrixX & eigenbasis, VectorX & perpendicular_velocity )
{
    int nos = spins.size();

    Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST,
         "  Calculate_Perpendicular_Velocity: calculate velocity matrix" );

    // Calculate the velocity matrix in the 3N-basis
    MatrixX velocity( 3 * nos, 3 * nos );
    Calculate_Dynamical_Matrix( spins, mu_s, hessian, velocity );

    Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST,
         "  Calculate_Perpendicular_Velocity: project velocity matrix" );

    // Project the velocity matrix into the 2N tangent space
    MatrixX velocity_projected( 2 * nos, 2 * nos );
    velocity_projected = basis.transpose() * velocity * basis;

    Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "  Calculate_Perpendicular_Velocity: calculate a" );

    // The velocity components orthogonal to the dividing surface
    perpendicular_velocity = eigenbasis.col( 0 ).transpose() * ( velocity_projected * eigenbasis );

    // std::cerr << "  Calculate_Perpendicular_Velocity: sorting" << std::endl;
    // std::sort(perpendicular_velocity.data(),perpendicular_velocity.data()+perpendicular_velocity.size());

    std::vector<std::string> block( 0 );
    for( int i = 0; i < 10; ++i )
        block.push_back( fmt::format( "  a[{}] = {}", i, perpendicular_velocity[i] ) );
    Log.SendBlock( Utility::Log_Level::Info, Utility::Log_Sender::HTST, block, -1, -1 );

    // std::cerr << "without units:" << std::endl;
    // for (int i=0; i<10; ++i)
    //     std::cerr << "  a[" << i << "] = " << perpendicular_velocity[i]/C::mu_B << std::endl;
}

void Calculate_Dynamical_Matrix(
    const vectorfield & spins, const scalarfield & mu_s, const MatrixX & hessian, MatrixX & velocity )
{
    velocity.setZero();
    int nos = spins.size();

    for( int i = 0; i < nos; ++i )
    {
        Vector3 beff{ 0, 0, 0 };

        for( int j = 0; j < nos; ++j )
        {
            velocity( 3 * i, 3 * j )
                = spins[i][1] * hessian( 3 * i + 2, 3 * j ) - spins[i][2] * hessian( 3 * i + 1, 3 * j );
            velocity( 3 * i, 3 * j + 1 )
                = spins[i][1] * hessian( 3 * i + 2, 3 * j + 1 ) - spins[i][2] * hessian( 3 * i + 1, 3 * j + 1 );
            velocity( 3 * i, 3 * j + 2 )
                = spins[i][1] * hessian( 3 * i + 2, 3 * j + 2 ) - spins[i][2] * hessian( 3 * i + 1, 3 * j + 2 );

            velocity( 3 * i + 1, 3 * j )
                = spins[i][2] * hessian( 3 * i, 3 * j ) - spins[i][0] * hessian( 3 * i + 2, 3 * j );
            velocity( 3 * i + 1, 3 * j + 1 )
                = spins[i][2] * hessian( 3 * i, 3 * j + 1 ) - spins[i][0] * hessian( 3 * i + 2, 3 * j + 1 );
            velocity( 3 * i + 1, 3 * j + 2 )
                = spins[i][2] * hessian( 3 * i, 3 * j + 2 ) - spins[i][0] * hessian( 3 * i + 2, 3 * j + 2 );

            velocity( 3 * i + 2, 3 * j )
                = spins[i][0] * hessian( 3 * i + 1, 3 * j ) - spins[i][1] * hessian( 3 * i, 3 * j );
            velocity( 3 * i + 2, 3 * j + 1 )
                = spins[i][0] * hessian( 3 * i + 1, 3 * j + 1 ) - spins[i][1] * hessian( 3 * i, 3 * j + 1 );
            velocity( 3 * i + 2, 3 * j + 2 )
                = spins[i][0] * hessian( 3 * i + 1, 3 * j + 2 ) - spins[i][1] * hessian( 3 * i, 3 * j + 2 );

            beff -= hessian.block<3, 3>( 3 * i, 3 * j ) * spins[j];
        }

        velocity( 3 * i, 3 * i + 1 ) -= beff[2];
        velocity( 3 * i, 3 * i + 2 ) += beff[1];
        velocity( 3 * i + 1, 3 * i ) += beff[2];
        velocity( 3 * i + 1, 3 * i + 2 ) -= beff[0];
        velocity( 3 * i + 2, 3 * i ) -= beff[1];
        velocity( 3 * i + 2, 3 * i + 1 ) += beff[0];

        velocity.row( 3 * i ) /= mu_s[i];
        velocity.row( 3 * i + 1 ) /= mu_s[i];
        velocity.row( 3 * i + 2 ) /= mu_s[i];
    }
}

void hessian_bordered_3N(
    const vectorfield & image, const vectorfield & gradient, const MatrixX & hessian, MatrixX & hessian_out )
{
    // Calculates a 3Nx3N matrix in the bordered Hessian approach and transforms it into the tangent basis,
    // making the result a 2Nx2N matrix. The bordered Hessian's Lagrange multipliers assume a local extremum.

    int nos     = image.size();
    hessian_out = hessian;

    VectorX lambda( nos );
    for( int i = 0; i < nos; ++i )
        lambda[i] = image[i].normalized().dot( gradient[i] );

    for( int i = 0; i < nos; ++i )
    {
        for( int j = 0; j < 3; ++j )
        {
            hessian_out( 3 * i + j, 3 * i + j ) -= lambda[i];
        }
    }
}

// NOTE WE ASSUME A SELFADJOINT MATRIX
void Eigen_Decomposition( const MatrixX & matrix, VectorX & evalues, MatrixX & evectors )
{
    // Create a Spectra solver
    Eigen::SelfAdjointEigenSolver<MatrixX> matrix_solver( matrix );
    evalues  = matrix_solver.eigenvalues().real();
    evectors = matrix_solver.eigenvectors().real();
}

void Eigen_Decomposition_Spectra(
    int nos, const MatrixX & matrix, VectorX & evalues, MatrixX & evectors, int n_decompose = 1 )
{
    int n_steps = std::max( 2, nos );

    //      Create a Spectra solver
    Spectra::DenseGenMatProd<scalar> op( matrix );
    Spectra::GenEigsSolver<scalar, Spectra::SMALLEST_REAL, Spectra::DenseGenMatProd<scalar>> matrix_spectrum(
        &op, n_decompose, n_steps );
    matrix_spectrum.init();

    //      Compute the specified spectrum
    int nconv = matrix_spectrum.compute();

    if( matrix_spectrum.info() == Spectra::SUCCESSFUL )
    {
        evalues  = matrix_spectrum.eigenvalues().real();
        evectors = matrix_spectrum.eigenvectors().real();
        // Eigen::Ref<VectorX> evec = evectors.col(0);
    }
    else
    {
        Log( Utility::Log_Level::Error, Utility::Log_Sender::All, "Failed to calculate eigenvectors of the Matrix!" );
        evalues.setZero();
        evectors.setZero();
    }
}

void Geodesic_Eigen_Decomposition(
    const vectorfield & image, const vectorfield & gradient, const MatrixX & hessian, MatrixX & hessian_geodesic_3N,
    MatrixX & hessian_geodesic_2N, VectorX & eigenvalues, MatrixX & eigenvectors )
{
    Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "---------- Geodesic Eigen Decomposition" );

    int nos = image.size();

    // Calculate geodesic Hessian in 3N-representation
    hessian_geodesic_3N = MatrixX::Zero( 3 * nos, 3 * nos );
    hessian_bordered_3N( image, gradient, hessian, hessian_geodesic_3N );

    // Transform into geodesic Hessian
    Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Transforming Hessian into geodesic Hessian..." );
    hessian_geodesic_2N = MatrixX::Zero( 2 * nos, 2 * nos );
    // Manifoldmath::hessian_bordered(image, gradient, hessian, hessian_geodesic_2N);
    // Manifoldmath::hessian_projected(image, gradient, hessian, hessian_geodesic_2N);
    // Manifoldmath::hessian_weingarten(image, gradient, hessian, hessian_geodesic_2N);
    // Manifoldmath::hessian_spherical(image, gradient, hessian, hessian_geodesic_2N);
    // Manifoldmath::hessian_covariant(image, gradient, hessian, hessian_geodesic_2N);

    // Do this manually
    MatrixX basis = MatrixX::Zero( 3 * nos, 2 * nos );
    Manifoldmath::tangent_basis_spherical( image, basis );
    // Manifoldmath::tangent_basis(image, basis);
    hessian_geodesic_2N = basis.transpose() * hessian_geodesic_3N * basis;

    // Calculate full eigenspectrum
    Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Calculation of full eigenspectrum..." );
    // std::cerr << hessian_geodesic_2N.cols() << "   " << hessian_geodesic_2N.rows() << std::endl;
    eigenvalues  = VectorX::Zero( 2 * nos );
    eigenvectors = MatrixX::Zero( 2 * nos, 2 * nos );
    Eigen_Decomposition( hessian_geodesic_2N, eigenvalues, eigenvectors );
    // Eigen_Decomposition_Spectra(hessian_geodesic_2N, eigenvalues, eigenvectors);

    Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "---------- Geodesic Eigen Decomposition Done" );
}

} // end namespace HTST
} // end namespace Engine

#endif