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
#include <engine/StateType.hpp>

#include "catch.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>

#include <iomanip>
#include <iostream>
#include <sstream>

using Catch::Matchers::WithinAbs;

using Engine::StateType;
using Engine::Field;
using Engine::get;

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

TEST_CASE( "Ensure that Hamiltonian is really just an aggregator", "[aggregation]" )
{
    // Hamiltonians to be tested
    std::vector<const char *> hamiltonian_input_files{
        "core/test/input/fd_gaussian.cfg",
        "core/test/input/fd_pairs.cfg",
        "core/test/input/fd_neighbours.cfg",
    };

    for( const auto * input_file : hamiltonian_input_files )
    {
        INFO( " Testing" << input_file );

        auto state = std::shared_ptr<State>( State_Setup( input_file ), State_Delete );
        Configuration_Random( state.get() );
        const auto & spins = *state->active_image->state;
        auto & hamiltonian = state->active_image->hamiltonian;
        auto nos           = get<Field::Spin>( spins ).size();

        if( hamiltonian->active_count() == 0 )
        {
            CAPTURE( fmt::format( " Warning: input file \"{}\" didn't specify any interaction to test.", input_file ) );
        }

        auto active_interactions = hamiltonian->active_interactions();
        auto aggregator          = [&active_interactions]( const auto init, const auto & f )
        { return std::accumulate( std::begin( active_interactions ), std::end( active_interactions ), init, f ); };

        scalar energy_hamiltonian = hamiltonian->Energy( spins );
        scalar energy_aggregated  = aggregator(
            0.0,
            [&spins]( const scalar v, const auto & interaction ) -> scalar
            { return v + interaction->Energy( spins ); } );

        INFO( "Hamiltonian::Energy" )
        INFO( "[total], epsilon = " << epsilon_2 << "\n" );
        INFO( "Energy (Hamiltonian) = " << energy_hamiltonian << "\n" );
        INFO( "Energy (aggregated)  = " << energy_aggregated << "\n" );
        REQUIRE_THAT( energy_hamiltonian, WithinAbs( energy_aggregated, epsilon_2 ) );

        scalarfield energy_per_spin_hamiltonian{}; // resize and clear should be handled by the hamiltonian
        hamiltonian->Energy_per_Spin( spins, energy_per_spin_hamiltonian );
        scalarfield energy_per_spin_aggregated = aggregator(
            scalarfield( nos, 0 ),
            [&spins]( const scalarfield & v, const auto & interaction ) -> scalarfield
            {
                const auto nos = get<Field::Spin>( spins ).size();
                auto energy_per_spin = scalarfield( nos, 0 );
                interaction->Energy_per_Spin( spins, energy_per_spin );
#pragma omp parallel for
                for( std::size_t i = 0; i < nos; ++i )
                    energy_per_spin[i] += v[i];

                return energy_per_spin;
            } );

        for( int i = 0; i < state->nos; i++ )
        {
            INFO( "Hamiltonian::Energy_per_Spin" )
            INFO( "i = " << i << ", epsilon = " << epsilon_2 << "\n" );
            INFO( "Energy (Hamiltonian)   = " << energy_per_spin_hamiltonian[i] << "\n" );
            INFO( "Energy (aggregated) = " << energy_per_spin_aggregated[i] << "\n" );
            REQUIRE_THAT( energy_per_spin_hamiltonian[i], WithinAbs( energy_per_spin_aggregated[i], epsilon_2 ) );
        }

        vectorfield gradient_hamiltonian{}; // resize and clear should be handled by the hamiltonian
        hamiltonian->Gradient( spins, gradient_hamiltonian );
        vectorfield gradient_aggregated = aggregator(
            vectorfield( nos, Vector3::Zero() ),
            [&spins]( const vectorfield & v, const auto & interaction ) -> vectorfield
            {
                auto gradient = vectorfield( get<Field::Spin>( spins ).size(), Vector3::Zero() );
                interaction->Gradient( spins, gradient );
                Engine::Vectormath::add_c_a( 1.0, v, gradient );
                return gradient;
            } );

        for( int i = 0; i < state->nos; i++ )
        {
            INFO( "Hamiltonian::Gradient" )
            INFO( "i = " << i << ", epsilon = " << epsilon_2 << "\n" );
            INFO( "Gradient (Hamiltonian)   = " << gradient_hamiltonian[i] << "\n" );
            INFO( "Gradient (aggregated) = " << gradient_aggregated[i] << "\n" );
            REQUIRE( gradient_hamiltonian[i].isApprox( gradient_aggregated[i], epsilon_2 ) );
        }

        scalar energy_combined_hamiltonian = 0;
        vectorfield gradient_combined_hamiltonian{};
        hamiltonian->Gradient_and_Energy( spins, gradient_combined_hamiltonian, energy_combined_hamiltonian );

        for( int i = 0; i < state->nos; i++ )
        {
            INFO( "Hamiltonian::Gradient_and_Energy" )
            INFO( "i = " << i << ", epsilon = " << epsilon_2 << "\n" );
            INFO( "Gradient (combined)   = " << gradient_combined_hamiltonian[i] << "\n" );
            INFO( "Gradient (aggregated) = " << gradient_aggregated[i] << "\n" );
            REQUIRE( gradient_combined_hamiltonian[i].isApprox( gradient_aggregated[i], epsilon_2 ) );
        }

        INFO( "Hamiltonian::Gradient_and_Energy" )
        INFO( "[total], epsilon = " << epsilon_2 << "\n" );
        INFO( "Energy (combined)   = " << energy_combined_hamiltonian << "\n" );
        INFO( "Energy (aggregated) = " << energy_aggregated << "\n" );
        REQUIRE_THAT( energy_combined_hamiltonian, WithinAbs( energy_aggregated, epsilon_2 ) );
    }
}
