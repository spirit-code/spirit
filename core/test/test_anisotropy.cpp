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

#include <Eigen/Core>
#include <Eigen/Dense>
#include <catch.hpp>

using Catch::Matchers::WithinAbs;

// Reduce required precision if float accuracy
#ifdef SPIRIT_SCALAR_TYPE_DOUBLE
constexpr scalar epsilon_rough = 1e-12;
#else
constexpr scalar epsilon_rough = 1e-1;
#endif

TEST_CASE( "Uniaxial nisotropy", "[anisotropy]" )
{
    auto state = std::shared_ptr<State>( State_Setup(), State_Delete );

    // Set the uniaxial anisotropy
    scalar init_magnitude = 0.1;
    scalar init_normal[3] = { 0.0, 0.0, 1.0 };
    Hamiltonian_Set_Anisotropy( state.get(), init_magnitude, init_normal );
    // Set the cubic anisotropy to zero
    Hamiltonian_Set_Cubic_Anisotropy( state.get(), 0 );

    SECTION( "Get after set should return the previously set value" )
    {
        scalar magnitude{};
        scalar normal[3]{};
        Hamiltonian_Get_Anisotropy( state.get(), &magnitude, normal );

        REQUIRE_THAT( magnitude, WithinAbs( init_magnitude, 1e-12 ) );
        REQUIRE_THAT( normal[0], WithinAbs( init_normal[0], 1e-12 ) );
        REQUIRE_THAT( normal[1], WithinAbs( init_normal[1], 1e-12 ) );
        REQUIRE_THAT( normal[2], WithinAbs( init_normal[2], 1e-12 ) );
    }

    SECTION( "Total energies for different orientations should match expected values" )
    {
        vectorfield spins( state->nos );

        for( auto & spin : spins )
            spin = { 1.0, 0.0, 0.0 };
        scalar energy_x = state->active_image->hamiltonian->Energy( spins );

        // X and Z orientations energies should differ by NOS*init_magnitude
        for( auto & spin : spins )
            spin = { 0.0, 0.0, 1.0 };
        scalar energy_z = state->active_image->hamiltonian->Energy( spins );
        REQUIRE_THAT( energy_x - energy_z, WithinAbs( init_magnitude * state->nos, epsilon_rough ) );

        // X and XY orientations energies should have equal energies
        scalar sqrt2_2 = std::sqrt( 2 ) / 2;
        for( auto & spin : spins )
            spin = { sqrt2_2, sqrt2_2, 0.0 };
        scalar energy_xy = state->active_image->hamiltonian->Energy( spins );
        REQUIRE_THAT( energy_x - energy_xy, WithinAbs( 0, 1e-12 ) );
    }

    SECTION( "All individual energy gradients should match the expected value" )
    {
        vectorfield spins( state->nos, { 0.0, 0.0, 1.0 } );

        auto gradients = vectorfield( state->nos );
        state->active_image->hamiltonian->Gradient( spins, gradients );

        Vector3 gradient_expected{ 0.0, 0.0, scalar( -2.0 * init_magnitude ) };
        for( int idx = 0; idx < state->nos; idx++ )
        {
            INFO(
                "i = " << idx << ", Gradient = " << gradients[idx].transpose() << " was expected to be "
                       << gradient_expected.transpose() << "\n" );
            REQUIRE( gradients[idx].isApprox( gradient_expected, 1e-12 ) );
        }
    }
}

TEST_CASE( "Cubic anisotropy", "[anisotropy]" )
{
    auto state = std::shared_ptr<State>( State_Setup(), State_Delete );
    vectorfield spins( state->nos );

    // Set uniaxial anisotropy to zero
    scalar init_normal_uniaxial[3] = { 0.0, 0.0, 1.0 };
    Hamiltonian_Set_Anisotropy( state.get(), 0.0, init_normal_uniaxial );
    // Set the cubic anisotropy
    scalar init_magnitude = 0.2;
    Hamiltonian_Set_Cubic_Anisotropy( state.get(), init_magnitude );

    SECTION( "Get after set should return the previously set value" )
    {
        scalar magnitude{};
        Hamiltonian_Get_Cubic_Anisotropy( state.get(), &magnitude );
        REQUIRE_THAT( magnitude, WithinAbs( init_magnitude, 1e-12 ) );
    }

    SECTION( "Total energies for different orientations should match expected values" )
    {
        scalar sqrt2_2 = std::sqrt( 2 ) / 2;
        for( auto & spin : spins )
            spin = { sqrt2_2, sqrt2_2, 0.0 };
        scalar energy_xy = state->active_image->hamiltonian->Energy( spins );

        // X and XY orientations energies should differ by NOS*init_magnitude/4
        for( auto & spin : spins )
            spin = { 1.0, 0.0, 0.0 };
        scalar energy_x = state->active_image->hamiltonian->Energy( spins );
        REQUIRE_THAT( energy_x - energy_xy, WithinAbs( -init_magnitude / 4 * state->nos, epsilon_rough ) );

        // Y and XY orientations energies should differ by NOS*init_magnitude/4
        for( auto & spin : spins )
            spin = { 0.0, 1.0, 0.0 };
        scalar energy_y = state->active_image->hamiltonian->Energy( spins );
        REQUIRE_THAT( energy_y - energy_xy, WithinAbs( -init_magnitude / 4 * state->nos, epsilon_rough ) );

        // Y and Z orientations should have equal energies
        for( auto & spin : spins )
            spin = { 0.0, 0.0, 1.0 };
        scalar energy_z = state->active_image->hamiltonian->Energy( spins );
        REQUIRE_THAT( energy_y - energy_z, WithinAbs( 0, 1e-12 ) );
    }

    SECTION( "All individual energy gradients should match the expected value" )
    {
        for( auto & spin : spins )
            spin = { 0.0, 0.0, 1.0 };
        auto gradients = vectorfield( state->nos );
        state->active_image->hamiltonian->Gradient( spins, gradients );

        Vector3 gradient_expected{ 0.0, 0.0, scalar( -2.0 * init_magnitude ) };

        for( int idx = 0; idx < state->nos; idx++ )
        {
            INFO(
                "i = " << idx << ", Gradient = " << gradients[idx].transpose() << " was expected to be "
                       << gradient_expected.transpose() << "\n" );
            REQUIRE( gradients[idx].isApprox( gradient_expected, 1e-12 ) );
        }
    }

    SECTION( "Test the Gradient_and_Energy function for both uniaxial and cubic anisotropy" )
    {
        Hamiltonian_Set_Anisotropy( state.get(), 0.1, init_normal_uniaxial );

        for( auto & spin : spins )
            spin = { 0.0, 0.0, 1.0 };

        // Direct energy calculation and energy calculated from gradient should be equal
        auto gradients_a = vectorfield( state->nos );
        scalar energy_from_gradient{};
        state->active_image->hamiltonian->Gradient_and_Energy( spins, gradients_a, energy_from_gradient );
        scalar energy_direct = state->active_image->hamiltonian->Energy( spins );
        REQUIRE_THAT( energy_from_gradient, WithinAbs( energy_direct, 1e-12 ) );

        // Direct gradient calculation and gradient out of gradient-and-energy calculation should be equal
        auto gradients_b = vectorfield( state->nos );
        state->active_image->hamiltonian->Gradient( spins, gradients_b );
        for( int idx = 0; idx < state->nos; ++idx )
        {
            INFO(
                "i = " << idx << ", Gradient from `Gradient_and_Energy` = " << gradients_a[idx].transpose()
                       << " was expected to be equal to directly calculated gradient " << gradients_b[idx].transpose()
                       << "\n" );
            REQUIRE( gradients_a[idx].isApprox( gradients_b[idx], 1e-12 ) );
        }
    }
}
