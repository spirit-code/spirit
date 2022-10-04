#include <Spirit/Configurations.h>
#include <Spirit/Constants.h>
#include <Spirit/Geometry.h>
#include <Spirit/Hamiltonian.h>
#include <Spirit/Parameters_LLG.h>
#include <Spirit/Simulation.h>
#include <Spirit/State.h>
#include <Spirit/System.h>
#include <Spirit/Version.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <catch.hpp>
#include <data/State.hpp>
#include <iomanip>
#include <iostream>
#include <sstream>

TEST_CASE( "Larmor Precession", "[physics]" )
{
    // Input file
    auto inputfile = "core/test/input/physics_larmor.cfg";

    // Create State
    auto state = std::shared_ptr<State>( State_Setup( inputfile ), State_Delete );

    // Solvers to be tested
    std::vector<int> solvers{ Solver_Heun, Solver_Depondt, Solver_SIB, Solver_RungeKutta4 };

    // Set up one the initial direction of the spin
    float init_direction[3] = { 1., 0., 0. };            // vec parallel to x-axis
    Configuration_Domain( state.get(), init_direction ); // set spin parallel to x-axis

    // Assure that the initial direction is set
    // (note: this pointer will stay valid throughout this test)
    auto direction = System_Get_Spin_Directions( state.get() );
    REQUIRE( direction[0] == 1 );
    REQUIRE( direction[1] == 0 );
    REQUIRE( direction[2] == 0 );

    // Make sure that mu_s is the same as the one define in input file
    float mu_s;
    Geometry_Get_mu_s( state.get(), &mu_s );
    REQUIRE( mu_s == 2 );

    // Get the magnitude of the magnetic field ( it has only z-axis component )
    float B_mag;
    float normal[3];
    Hamiltonian_Get_Field( state.get(), &B_mag, normal );

    // Get time step of method
    scalar damping = 0.3;
    float tstep    = Parameters_LLG_Get_Time_Step( state.get() );
    Parameters_LLG_Set_Damping( state.get(), damping );

    scalar dtg = tstep * Constants_gamma() / ( 1.0 + damping * damping );

    for( auto opt : solvers )
    {
        // Set spin parallel to x-axis
        Configuration_Domain( state.get(), init_direction );
        Simulation_LLG_Start( state.get(), opt, -1, -1, true );

        for( int i = 0; i < 100; i++ )
        {
            INFO( "solver " << opt << " failed spin trajectory test at iteration " << i );

            // A single iteration
            Simulation_SingleShot( state.get() );

            // Expected spin orientation
            scalar phi_expected = dtg * ( i + 1 ) * B_mag;
            scalar sz_expected  = std::tanh( damping * dtg * ( i + 1 ) * B_mag );
            scalar rxy_expected = std::sqrt( 1 - sz_expected * sz_expected );
            scalar sx_expected  = std::cos( phi_expected ) * rxy_expected;

            REQUIRE( Approx( direction[0] ) == sx_expected );
            REQUIRE( Approx( direction[2] ) == sz_expected );
        }

        Simulation_Stop( state.get() );
    }
}

TEST_CASE( "Finite Differences", "[physics]" )
{
    // Hamiltonians to be tested
    std::vector<const char *> hamiltonians{ "core/test/input/fd_pairs.cfg" };
    //"core/test/input/fd_neighbours",
    //"core/test/input/fd_gaussian.cfg"};

    // Reduce precision if float accuracy
    double epsilon_apprx = 1e-11;
    if( strcmp( Spirit_Scalar_Type(), "float" ) == 0 )
    {
        WARN( "Detected single precision calculation. Reducing precision requirements." );
        epsilon_apprx = 1e-4;
    }

    for( auto ham : hamiltonians )
    {
        INFO( " Testing " << ham );

        // create state
        auto state = std::shared_ptr<State>( State_Setup( ham ), State_Delete );

        Configuration_Random( state.get() );

        auto & vf    = *state->active_image->spins;
        auto grad    = vectorfield( state->nos );
        auto grad_fd = vectorfield( state->nos );

        state->active_image->hamiltonian->Gradient_FD( vf, grad_fd );
        state->active_image->hamiltonian->Gradient( vf, grad );

        for( int i = 0; i < state->nos; i++ )
        {
            INFO( "i = " << i << "\n" );
            INFO( "Gradient (FD) = " << grad_fd[i].transpose() << "\n" );
            INFO( "Gradient      = " << grad[i].transpose() << "\n" );
            REQUIRE( grad_fd[i].isApprox( grad[i], epsilon_apprx ) );
        }

        auto hessian    = MatrixX( 3 * state->nos, 3 * state->nos );
        auto hessian_fd = MatrixX( 3 * state->nos, 3 * state->nos );

        state->active_image->hamiltonian->Hessian_FD( vf, hessian_fd );
        state->active_image->hamiltonian->Hessian( vf, hessian );

        INFO( "Hessian (FD) = " << hessian_fd << "\n" );
        INFO( "Hessian      = " << hessian << "\n" );
        REQUIRE( hessian_fd.isApprox( hessian, epsilon_apprx ) );
    }
}

TEST_CASE( "Dipole-Dipole Interaction", "[physics]" )
{
    // cfg where only ddi is enabled
    auto state = std::shared_ptr<State>( State_Setup( "core/test/input/physics_ddi.cfg" ), State_Delete );

    Configuration_Random( state.get() );

    auto & spins = *state->active_image->spins;

    auto grad_fft    = vectorfield( state->nos );
    auto grad_direct = vectorfield( state->nos );

    state->active_image->hamiltonian->Gradient( spins, grad_fft );
    auto energy_fft = state->active_image->hamiltonian->Energy( spins );

    auto n_periodic_images = std::vector<int>{ 4, 4, 4 };
    Hamiltonian_Set_DDI( state.get(), SPIRIT_DDI_METHOD_CUTOFF, n_periodic_images.data(), -1 );

    state->active_image->hamiltonian->Gradient( spins, grad_direct );
    auto energy_direct = state->active_image->hamiltonian->Energy( spins );

    for( int i = 0; i < state->nos; i++ )
    {
        INFO( "Failed DDI-Gradient comparison at i = " << i );
        INFO( "Gradient (FFT):" )
        INFO( grad_fft[i] )
        INFO( "Gradient (Direct):" )
        INFO( grad_direct[i] )
        REQUIRE( grad_fft[i].isApprox( grad_direct[i] ) );
    }
    INFO( "Failed energy comparison test!" )
    INFO( "Energy (Direct) = " << energy_direct << "\n" );
    INFO( "Energy (FFT)    = " << energy_fft << "\n" );
    REQUIRE( Approx( energy_fft ) == energy_direct );
}
