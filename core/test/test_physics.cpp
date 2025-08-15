#include <Spirit/Configurations.h>
#include <Spirit/Constants.h>
#include <Spirit/Geometry.h>
#include <Spirit/Hamiltonian.h>
#include <Spirit/Parameters_LLG.h>
#include <Spirit/Quantities.h>
#include <Spirit/Simulation.h>
#include <Spirit/State.h>
#include <Spirit/System.h>
#include <Spirit/Version.h>
#include <data/State.hpp>
#include <engine/Vectormath.hpp>
#include <engine/spin/Method_Solver.hpp>
#include <utility/Enum.hpp>

#include "catch.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>

using Catch::Matchers::WithinAbs;
namespace Enum = Utility::Enum;

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

TEST_CASE( "Dynamics solvers should follow Larmor precession", "[physics]" )
{
    using Engine::Spin::Solver;
    constexpr auto input_file = "core/test/input/physics_larmor.toml";
    std::vector<Solver> solvers{
        Solver::Heun,
        Solver::Depondt,
        Solver::SIB,
        Solver::RungeKutta4,
    };

    // Create State
    auto state = std::shared_ptr<State>( State_Setup( input_file ), State_Delete );
    REQUIRE( state != nullptr );
    REQUIRE( !state->config_file.empty() );

    // Set up one the initial direction of the spin
    scalar init_direction[3] = { 1., 0., 0. };           // vec parallel to x-axis
    Configuration_Domain( state.get(), init_direction ); // set spin parallel to x-axis

    // Assure that the initial direction is set
    // (note: this pointer will stay valid throughout this test)
    const auto * direction = System_Get_Spin_Directions( state.get() );
    REQUIRE( direction[0] == 1 );
    REQUIRE( direction[1] == 0 );
    REQUIRE( direction[2] == 0 );

    // Make sure that mu_s is the same as the one define in input file
    scalar mu_s{};
    Geometry_Get_mu_s( state.get(), &mu_s );
    REQUIRE( mu_s == 2 );

    // Get the magnitude of the magnetic field (it has only z-axis component)
    scalar B_mag{};
    scalar normal[3]{ 0, 0, 1 };
    Hamiltonian_Get_Field( state.get(), &B_mag, normal );

    // Get time step of method
    scalar damping = 0.3;
    scalar tstep   = Parameters_LLG_Get_Time_Step( state.get() );
    Parameters_LLG_Set_Damping( state.get(), damping );

    scalar dtg = tstep * Constants_gamma() / ( 1.0 + damping * damping );

    for( auto solver : solvers )
    {
        // Set spin parallel to x-axis
        Configuration_Domain( state.get(), init_direction );
        Simulation_LLG_Start( state.get(), static_cast<int>( solver ), -1, -1, true );

        for( int i = 0; i < 100; i++ )
        {
            INFO( fmt::format(
                "Solver \"{}: {}\" failed spin trajectory test at iteration {}", static_cast<int>( solver ),
                Enum::name( solver ), i ) );

            // A single iteration
            Simulation_SingleShot( state.get() );

            // Expected spin orientation
            scalar phi_expected = dtg * ( i + 1 ) * B_mag;
            scalar sz_expected  = std::tanh( damping * dtg * ( i + 1 ) * B_mag );
            scalar rxy_expected = std::sqrt( 1 - sz_expected * sz_expected );
            scalar sx_expected  = std::cos( phi_expected ) * rxy_expected;

            // TODO: why is precision so low for Heun and SIB solvers? Other solvers manage ~1e-10
            REQUIRE_THAT( direction[0], WithinAbs( sx_expected, epsilon_5 ) );
            REQUIRE_THAT( direction[2], WithinAbs( sz_expected, 1e-6 ) );
        }

        Simulation_Stop( state.get() );
    }
}

TEST_CASE(
    "RK4 should not dephase spins while they precess with active exchange interaction and open boundary conditions",
    "[physics]" )
{
    using Engine::Spin::Solver;
    constexpr auto input_file = "core/test/input/physics_dephasing.toml";
    std::vector<Solver> solvers{
        Solver::RungeKutta4,
    };

    // Create State
    auto state = std::shared_ptr<State>( State_Setup( input_file ), State_Delete );
    REQUIRE( state != nullptr );
    REQUIRE( !state->config_file.empty() );

    // Set up one the initial direction of the spin
    static constexpr scalar theta  = 0.1 * Utility::Constants::Pi;
    const scalar init_direction[3] = { sin( theta ), 0., cos( theta ) };
    Configuration_Domain( state.get(), init_direction );

    const int nos = System_Get_NOS( state.get() );

    // Assure that the initial direction is set
    // (note: this pointer will stay valid throughout this test)
    const auto * direction = System_Get_Spin_Directions( state.get() );
    for( int i = 0; i < 3 * nos; i += 3 )
    {
        REQUIRE_THAT( direction[i + 0], WithinAbs( sin( theta ), epsilon_4 ) );
        REQUIRE_THAT( direction[i + 1], WithinAbs( 0., epsilon_4 ) );
        REQUIRE_THAT( direction[i + 2], WithinAbs( cos( theta ), epsilon_4 ) );
    }

    // Make sure that mu_s is the same as the one define in input file
    scalar mu_s{};
    Geometry_Get_mu_s( state.get(), &mu_s );
    REQUIRE( mu_s == 2 );

    // Set the magnitude of the magnetic field (it has only z-axis component)
    const scalar B_mag = 0.0;
    const scalar normal[3]{ 0, 0, 1 };
    Hamiltonian_Set_Field( state.get(), B_mag, normal );

    // Set the magnitude of the anisotropy
    const scalar ani_mag = 1.0;
    const scalar ani_normal[3]{ 0, 0, 1 };
    Hamiltonian_Set_Anisotropy( state.get(), ani_mag, ani_normal );

    // Set the magnitude of the exchange interaction
    const scalar exchange_mag = 50;
    Hamiltonian_Set_Exchange( state.get(), 1, &exchange_mag );

    const Vector3 n0{ init_direction[0], init_direction[1], init_direction[2] };
    const Vector3 K0{ ani_normal[0], ani_normal[1], ani_normal[2] };

    // Set zero damping
    const scalar damping = 0.0;
    Parameters_LLG_Set_Damping( state.get(), damping );

    // absolute value of the std of the magnetization
    auto stdev = [nos, direction, state = state.get(), prefactor = 1.0 / static_cast<scalar>( std::max( 1, nos - 1 ) )]
    {
        scalar average[3] = { 0, 0, 0 };
        Quantity_Get_Average_Spin( state, average );

        scalar variance = 0;
        for( auto j = 0; j < 3 * nos; j += 3 )
        {
            for( auto k = 0; k < 3; ++k )
            {
                const scalar diff = direction[j + k] - average[k];
                variance += diff * diff;
            }
        }
        return sqrt( prefactor * variance );
    };

    for( auto solver : solvers )
    {
        // Set spin parallel to x-axis
        Configuration_Domain( state.get(), init_direction );
        Simulation_LLG_Start( state.get(), static_cast<int>( solver ), -1, -1, /*singleshot=*/true );

        for( int i = 0; i < 1000; ++i )
        {
            INFO( fmt::format(
                "Solver \"{}: {}\" failed spin dephasing test at iteration {}", static_cast<int>( solver ),
                Enum::name( solver ), i ) );

            // A single iteration
            Simulation_SingleShot( state.get() );
            REQUIRE_THAT( stdev(), WithinAbs( 0, epsilon_4 ) );
        }

        Simulation_Stop( state.get() );
    }
}
