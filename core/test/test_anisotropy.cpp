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
#include <engine/Vectormath_Defines.hpp>
#include <utility/Constants.hpp>

#include "catch.hpp"
#include "matchers.hpp"
#include "utility.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>

#include <cmath>

// Reduce required precision if float accuracy
#ifdef SPIRIT_SCALAR_TYPE_DOUBLE
[[maybe_unused]] constexpr int digits_a = 12;
[[maybe_unused]] constexpr int digits_b = 4;
[[maybe_unused]] constexpr int digits_c = 1;
#else
[[maybe_unused]] constexpr int digits_a = 1;
[[maybe_unused]] constexpr int digits_b = 4;
[[maybe_unused]] constexpr int digits_c = 1;
#endif

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

namespace C = Utility::Constants;

using Catch::CustomMatchers::MapApprox;
using Catch::CustomMatchers::within_digits;
using Catch::Matchers::Equals;
using Catch::Matchers::WithinAbs;

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

        REQUIRE_THAT( magnitude, within_digits( init_magnitude, 12 ) );
        REQUIRE_THAT( normal[0], within_digits( init_normal[0], 12 ) );
        REQUIRE_THAT( normal[1], within_digits( init_normal[1], 12 ) );
        REQUIRE_THAT( normal[2], within_digits( init_normal[2], 12 ) );
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
        REQUIRE_THAT( energy_x - energy_z, within_digits( init_magnitude * state->nos, digits_a ) );

        // X and XY orientations energies should have equal energies
        scalar sqrt2_2 = std::sqrt( 2 ) / 2;
        for( auto & spin : spins )
            spin = { sqrt2_2, sqrt2_2, 0.0 };
        scalar energy_xy = state->active_image->hamiltonian->Energy( spins );
        REQUIRE_THAT( energy_x - energy_xy, within_digits( 0, 12 ) );
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
            REQUIRE( gradients[idx].isApprox( gradient_expected, epsilon_3 ) );
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
        REQUIRE_THAT( magnitude, within_digits( init_magnitude, 12 ) );
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
        REQUIRE_THAT( energy_x - energy_xy, within_digits( -init_magnitude / 4 * state->nos, digits_a ) );

        // Y and XY orientations energies should differ by NOS*init_magnitude/4
        for( auto & spin : spins )
            spin = { 0.0, 1.0, 0.0 };
        scalar energy_y = state->active_image->hamiltonian->Energy( spins );
        REQUIRE_THAT( energy_y - energy_xy, within_digits( -init_magnitude / 4 * state->nos, digits_a ) );

        // Y and Z orientations should have equal energies
        for( auto & spin : spins )
            spin = { 0.0, 0.0, 1.0 };
        scalar energy_z = state->active_image->hamiltonian->Energy( spins );
        REQUIRE_THAT( energy_y - energy_z, within_digits( 0, 12 ) );
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
            REQUIRE( gradients[idx].isApprox( gradient_expected, epsilon_3 ) );
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
        REQUIRE_THAT( energy_from_gradient, within_digits( energy_direct, 12 ) );

        // Direct gradient calculation and gradient out of gradient-and-energy calculation should be equal
        auto gradients_b = vectorfield( state->nos );
        state->active_image->hamiltonian->Gradient( spins, gradients_b );
        for( int idx = 0; idx < state->nos; ++idx )
        {
            INFO(
                "i = " << idx << ", Gradient from `Gradient_and_Energy` = " << gradients_a[idx].transpose()
                       << " was expected to be equal to directly calculated gradient " << gradients_b[idx].transpose()
                       << "\n" );
            REQUIRE( gradients_a[idx].isApprox( gradients_b[idx], epsilon_3 ) );
        }
    }
}

TEST_CASE( "Biaxial anisotropy", "[anisotropy]" )
{
    auto state = std::shared_ptr<State>( State_Setup(), State_Delete );

    int idx_image = -1, idx_chain = -1;
    auto [image, chain] = from_indices( state.get(), idx_image, idx_chain );

    auto * hamiltonian = image->hamiltonian.get();
    REQUIRE( hamiltonian != nullptr );

    auto interaction = hamiltonian->getInteraction<Engine::Spin::Interaction::Biaxial_Anisotropy>();
    REQUIRE( interaction.get() != nullptr );

    using exponents_t = std::array<unsigned int, 3>;
    static constexpr int init_n_terms{ 7 };
    static constexpr std::array<scalar, 3> init_primary{ 0, 0, 1 };
    static constexpr std::array<scalar, 3> init_secondary{ 1, 0, 0 };
    static constexpr std::array init_exponents{
        std::array{ 1u, 0u, 0u }, std::array{ 2u, 0u, 0u }, std::array{ 3u, 0u, 0u }, std::array{ 1u, 2u, 0u },
        std::array{ 0u, 4u, 0u }, std::array{ 3u, 2u, 0u }, std::array{ 3u, 4u, 0u },
    };
    static constexpr std::array<scalar, 7> init_magnitudes{ 3.0, 1.8, 0.9, -3.2, 3.2, -1.6, 1.6 };

    SECTION( "API: Get after set should return the previously set value" )
    {
        Hamiltonian_Set_Biaxial_Anisotropy(
            state.get(), init_magnitudes.data(), array_cast( init_exponents ), init_primary.data(),
            init_secondary.data(), init_n_terms );

        const int n_atoms = Hamiltonian_Get_Biaxial_Anisotropy_N_Atoms( state.get() );

        REQUIRE( n_atoms == hamiltonian->get_geometry().n_cell_atoms );

        const int n_terms = Hamiltonian_Get_Biaxial_Anisotropy_N_Terms( state.get() );

        REQUIRE( n_terms > 0 );

        std::vector<int> indices( n_atoms );
        std::vector<std::array<scalar, 3>> primary( n_atoms );
        std::vector<std::array<scalar, 3>> secondary( n_atoms );
        std::vector<int> site_p( n_atoms + 1 );
        std::vector<scalar> magnitudes( n_terms );
        std::vector exponents( n_terms, std::array{ 0, 0, 0 } );

        Hamiltonian_Get_Biaxial_Anisotropy(
            state.get(), indices.data(), array_cast( primary ), array_cast( secondary ),
            site_p.data(), n_atoms, magnitudes.data(), array_cast( exponents ), n_terms );

        REQUIRE( site_p[0] == 0 );
        for( int i = 0; i < n_atoms; ++i )
        {
            REQUIRE( site_p[i + 1] - site_p[i] == init_n_terms );
        }

        std::vector<int> indices_ref( n_atoms );
        std::iota( begin( indices_ref ), end( indices_ref ), 0 );

        REQUIRE_THAT( indices, Equals( indices_ref ) );

        for( const auto & k1 : primary )
            for( int i = 0; i < 3; ++i )
            {
                INFO( fmt::format( "Expected: [{}]", array_fmt( init_primary ) ) );
                INFO( fmt::format( "Actual:   [{}]", array_fmt( k1 ) ) );
                REQUIRE_THAT( k1[i], WithinAbs( init_primary[i], epsilon_3 ) );
            }

        for( const auto & k2 : secondary )
            for( int i = 0; i < 3; ++i )
            {
                INFO( fmt::format( "Expected: [{}]", array_fmt( init_secondary ) ) );
                INFO( fmt::format( "Actual:   [{}]", array_fmt( k2 ) ) );
                REQUIRE_THAT( k2[i], WithinAbs( init_secondary[i], epsilon_3 ) );
            }

        // use std::map to compare them, because the order of the polynomial terms need not be fixed
        using term_idx = std::tuple<int, int, int>;
        using term_map = std::map<term_idx, scalar>;
        auto make_polynomial
            = []( const int offset_begin, const int offset_end, const auto & exponents, const auto & magnitudes )
        {
            term_map polynomial{};
            for( int i = offset_begin; i < offset_end; ++i )
                polynomial.emplace(
                    std::make_tuple( exponents[i][0], exponents[i][1], exponents[i][2] ), magnitudes[i] );
            return polynomial;
        };

        const auto init_polynomial = make_polynomial( 0, init_n_terms, init_exponents, init_magnitudes );
        for( int i = 0; i < n_atoms; ++i )
        {
            const auto polynomial = make_polynomial( site_p[i], site_p[i + 1], exponents, magnitudes );
            REQUIRE_THAT( polynomial, MapApprox( init_polynomial, epsilon_3 ) );
        }
    }

    auto term_info = []( const auto &... terms )
    {
        return fmt::format( "{} term(s):\n", sizeof...( terms ) )
               + ( ...
                   + fmt::format(
                       "    n1={}, n2={}, n3={}, c={}\n", terms.n1, terms.n2, terms.n3, terms.coefficient ) );
    };

    SECTION( "Engine: Check results of the energy calculations" )
    {
        const auto test = [&state, &interaction,
                           &term_info]( const int idx, const scalar theta, const scalar phi, const auto &... terms )
        {
            auto make_spherical = []( const scalar theta, const scalar phi ) -> Vector3 {
                return { cos( phi ) * sin( theta ), sin( phi ) * sin( theta ), cos( theta ) };
            };

            static constexpr std::size_t N = sizeof...( terms );

            const std::array<scalar, N> magnitude{ terms.coefficient... };
            const std::array<exponents_t, N> exponents{ exponents_t{ { terms.n1, terms.n2, terms.n3 } }... };

            Hamiltonian_Set_Biaxial_Anisotropy(
                state.get(), magnitude.data(), array_cast( exponents ), init_primary.data(), init_secondary.data(), N );

            vectorfield spins( state->nos, make_spherical( theta, phi ) );
            scalarfield energy( state->nos, 0.0 );
            interaction->Energy_per_Spin( spins, energy );

            // reference energy
            const scalar ref_energy
                = ( ...
                    + ( terms.coefficient * pow( sin( theta ), 2 * terms.n1 + terms.n2 + terms.n3 )
                        * pow( cos( phi ), terms.n2 ) * pow( sin( phi ), terms.n3 ) ) );

            for( const auto & e : energy )
            {
                INFO( "Energy:" )
                INFO( "trial: " << idx << ", theta=" << theta << ", phi=" << phi );
                INFO( term_info( terms... ) );
                REQUIRE_THAT( e, WithinAbs( ref_energy, epsilon_3 ) );
            };

            INFO( "Total Energy:" )
            INFO( "trial: " << idx << ", theta=" << theta << ", phi=" << phi );
            INFO( term_info( terms... ) );
            REQUIRE_THAT( interaction->Energy( spins ), WithinAbs( state->nos * ref_energy, epsilon_5 ) );

            vectorfield gradient( state->nos, Vector3::Zero() );
            interaction->Gradient( spins, gradient );

            const auto k1 = Vector3{ init_primary[0], init_primary[1], init_primary[2] };
            const auto k2 = Vector3{ init_secondary[0], init_secondary[1], init_secondary[2] };
            const auto k3 = k1.cross( k2 );

            Vector3 ref_gradient
                = ( ... +
                    [s1 = cos( theta ), s2 = sin( theta ) * cos( phi ), s3 = sin( theta ) * sin( phi ), &k1, &k2,
                     &k3]( const auto & term )
                    {
                        Vector3 result{ 0, 0, 0 };
                        const scalar a = pow( s2, term.n2 );
                        const scalar b = pow( s3, term.n3 );
                        const scalar c = pow( 1 - s1 * s1, term.n1 );

                        const auto & [coefficient, n1, n2, n3] = term;
                        if( n1 > 0 )
                            result += k1 * ( coefficient * a * b * n1 * ( -2 * s1 * pow( 1 - s1 * s1, n1 - 1 ) ) );
                        if( n2 > 0 )
                            result += k2 * ( coefficient * b * c * n2 * pow( s2, n2 - 1 ) );
                        if( n3 > 0 )
                            result += k3 * ( coefficient * a * c * n3 * pow( s3, n3 - 1 ) );
                        return result;
                    }( terms ) );

            for( const auto & g : gradient )
            {
                for( std::size_t i = 0; i < 3; ++i )
                {
                    INFO( "trial: " << idx << ", theta=" << theta << ", phi=" << phi );
                    INFO( term_info( terms... ) );
                    INFO( "Gradient(expected): " << ref_gradient.transpose() );
                    INFO( "Gradient(actual):   " << g.transpose() );
                    REQUIRE( g.isApprox( ref_gradient, epsilon_3 ) );
                }
            }
        };

        auto rng         = std::mt19937( 3548368 );
        auto angle_theta = std::uniform_real_distribution<scalar>( 0, C::Pi );
        auto angle_phi   = std::uniform_real_distribution<scalar>( -2 * C::Pi, 2 * C::Pi );
        auto coeff       = std::uniform_real_distribution<scalar>( -10.0, 10.0 );
        auto exp         = std::uniform_int_distribution<unsigned int>( 0, 6 );

        for( int n = 0; n < 6; ++n )
        {
            const scalar theta = angle_theta( rng );
            const scalar phi   = angle_phi( rng );

            std::array terms{
                PolynomialTerm{ coeff( rng ), exp( rng ), exp( rng ), exp( rng ) },
                PolynomialTerm{ coeff( rng ), exp( rng ), exp( rng ), exp( rng ) },
            };

            for( const auto & term : terms )
                test( n, theta, phi, term );

            for( const auto & term_a : terms )
                for( const auto & term_b : terms )
                    test( n, theta, phi, term_a, term_b );
        }
    }
}
