#ifdef SPIRIT_ENABLE_PINNING
#include <Spirit/Configurations.h>
#include <Spirit/Constants.h>
#include <Spirit/Geometry.h>
#include <Spirit/Hamiltonian.h>
#include <Spirit/Parameters_LLG.h>
#include <Spirit/Simulation.h>
#include <Spirit/State.h>
#include <Spirit/System.h>
#include <Spirit/Version.h>
#include <data/State.hpp>
#include <engine/Vectormath.hpp>
#include <engine/spin/Method_Solver.hpp>

#include "catch.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>

using Catch::Matchers::WithinAbs;
namespace C = Utility::Constants;

// Reduce required precision if float accuracy
#ifdef SPIRIT_SCALAR_TYPE_DOUBLE
[[maybe_unused]] constexpr scalar epsilon_2 = 1e-10;
[[maybe_unused]] constexpr scalar epsilon_3 = 1e-12;
[[maybe_unused]] constexpr scalar epsilon_4 = 1e-12;
[[maybe_unused]] constexpr scalar epsilon_5 = 1e-6;
[[maybe_unused]] constexpr scalar epsilon_6 = 1e-7;
#else
[[maybe_unused]] constexpr scalar epsilon_2 = 1e-2;
[[maybe_unused]] constexpr scalar epsilon_3 = 1e-3;
[[maybe_unused]] constexpr scalar epsilon_4 = 1e-4;
[[maybe_unused]] constexpr scalar epsilon_5 = 1e-5;
[[maybe_unused]] constexpr scalar epsilon_6 = 1e-6;
#endif

TEST_CASE( "Dynamics solvers should follow Larmor precession with one pinned spin and Heisenber exchange", "[physics]" )
{
    using Engine::Spin::Solver;
    constexpr auto input_file = "core/test/input/physics_pinning.cfg";
    const std::vector<Solver> solvers{
        Solver::RungeKutta4,
        Solver::Heun,
        Solver::Depondt,
        Solver::SIB,
    };

    // Create State
    auto state = std::shared_ptr<State>( State_Setup( input_file ), State_Delete );

    const auto setup = []( auto * state )
    {
        static constexpr scalar r_cut_cylindrical = 0.1;
        // Set up one the initial direction of the pinned spin
        const scalar pos_pinned[3]       = { 0.5, 0., 0. };
        const scalar pinned_direction[3] = { 0., 0., 1. };
        Configuration_Domain( state, pinned_direction, pos_pinned, defaultRect, r_cut_cylindrical );
        Configuration_Set_Pinned( state, true, pos_pinned, defaultRect, r_cut_cylindrical );

        // Set up one the initial direction of the free spin
        const scalar pos_free[3]       = { -0.5, 0., 0. };
        const scalar init_direction[3] = { 1., 0., 0. }; // vec parallel to x-axis
        Configuration_Domain( state, init_direction, pos_free, defaultRect, r_cut_cylindrical );
        Configuration_Set_Pinned( state, false, pos_free, defaultRect, r_cut_cylindrical );
    };

    // Make sure that mu_s is the same as the one define in input file
    const scalar mu_s = 2;
    Geometry_Set_mu_s( state.get(), mu_s );

    // Set the magnitude of the magnetic field to zero
    scalar normal[3]{ 0, 0, 1 };
    Hamiltonian_Set_Field( state.get(), 0.0, normal );

    // Set the magnitude of the exchange coupling to 2*μ_B
    const scalar omega_L  = 1.0;
    const scalar exchange = omega_L * 2.0 * C::mu_B;
    Hamiltonian_Set_Exchange( state.get(), 1, &exchange );

    REQUIRE( state->active_image->hamiltonian->active_count() == 1 );

    setup( state.get() );
    // Assure that the initial direction is set
    // NOTE: this pointer will stay valid as long as the size of the geometry doesn't change
    auto * direction = System_Get_Spin_Directions( state.get() );
    REQUIRE( direction[0] == 1 );
    REQUIRE( direction[1] == 0 );
    REQUIRE( direction[2] == 0 );

    REQUIRE( direction[3] == 0 );
    REQUIRE( direction[4] == 0 );
    REQUIRE( direction[5] == 1 );

    // Get time step of method
    const scalar damping = 0.3;
    const scalar tstep   = Parameters_LLG_Get_Time_Step( state.get() );
    Parameters_LLG_Set_Damping( state.get(), damping );

    const scalar dtg = tstep * C::gamma / ( 1.0 + damping * damping );

    for( const auto solver : solvers )
    {
        // Set free spin parallel to x-axis and pinned spin parallel to z-axis
        setup( state.get() );
        Simulation_LLG_Start( state.get(), static_cast<int>( solver ), -1, -1, true );

        for( int i = 0; i < 100; i++ )
        {
            INFO( fmt::format(
                "Solver {}: \"{}\" failed spin trajectory test at iteration {}", static_cast<int>( solver ),
                name( solver ), i ) );

            // A single iteration
            Simulation_SingleShot( state.get() );

            // Expected spin orientation for the precessing free spin
            const scalar phi_expected = dtg * ( i + 1 ) * omega_L;
            const scalar sz_expected  = std::tanh( damping * phi_expected );
            const scalar rxy_expected = std::sqrt( 1 - sz_expected * sz_expected );
            const scalar sx_expected  = std::cos( phi_expected ) * rxy_expected;

            // pinned spin
            REQUIRE_THAT( direction[3], WithinAbs( 0.0, 1e-12 ) );
            REQUIRE_THAT( direction[5], WithinAbs( 1.0, 1e-12 ) );

            // free spin
            REQUIRE_THAT( direction[2], WithinAbs( sz_expected, 1e-6 ) );
            REQUIRE_THAT( direction[0], WithinAbs( sx_expected, epsilon_5 ) );
        }

        Simulation_Stop( state.get() );
    }
}

#endif
