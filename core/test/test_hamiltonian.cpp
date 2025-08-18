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
[[maybe_unused]] constexpr scalar epsilon_2 = 1e-3;
[[maybe_unused]] constexpr scalar epsilon_3 = 1e-4;
[[maybe_unused]] constexpr scalar epsilon_4 = 1e-5;
[[maybe_unused]] constexpr scalar epsilon_5 = 1e-6;
[[maybe_unused]] constexpr scalar epsilon_6 = 1e-7;
#else
[[maybe_unused]] constexpr scalar epsilon_2 = 1e-2;
[[maybe_unused]] constexpr scalar epsilon_3 = 1e-3;
[[maybe_unused]] constexpr scalar epsilon_4 = 1e-4;
[[maybe_unused]] constexpr scalar epsilon_5 = 1e-5;
[[maybe_unused]] constexpr scalar epsilon_6 = 1e-6;
#endif

// Hamiltonians to be tested
static constexpr std::array input_files{
    "core/test/input/fd_pairs.toml",
    // "core/test/input/fd_gaussian.toml",  // The Gaussian interaction test fails randomly
    "core/test/input/fd_quadruplet.toml",
    // These should be sufficent at some point
    "core/test/input/fd_neighbours.toml",
    "core/test/input/hamiltonian.toml",
};

TEST_CASE( "Finite difference and regular Hamiltonian should match", "[hamiltonian]" )
{
    for( const auto * input_file : input_files )
    {
        INFO( " Testing " << input_file );

        auto state = std::shared_ptr<State>( State_Setup( input_file ), State_Delete );
        REQUIRE( state != nullptr );
        REQUIRE( !state->config_file.empty() );
        Configuration_Random( state.get() );
        const auto & system_state = *state->active_image->state;
        auto & hamiltonian        = state->active_image->hamiltonian;

        // Compare gradients
        auto grad    = vectorfield( state->nos, Vector3::Zero() );
        auto grad_fd = vectorfield( state->nos, Vector3::Zero() );
        for( const auto & interaction : hamiltonian->active_interactions() )
        {
            Engine::Vectormath::fill( grad, Vector3::Zero() );
            Engine::Vectormath::fill( grad_fd, Vector3::Zero() );
            interaction->Gradient( system_state, grad );
            Engine::Vectormath::Gradient(
                system_state, grad_fd,
                [&interaction]( const auto & state ) -> scalar { return interaction->Energy( state ); } );
            INFO( "Interaction: " << interaction->Name() << "\n" );
            for( int i = 0; i < state->nos; i++ )
            {
                INFO( "i = " << i << ", epsilon = " << epsilon_2 << "\n" );
                INFO( "Gradient (FD) = " << grad_fd[i].transpose() << "\n" );
                INFO( "Gradient      = " << grad[i].transpose() << "\n" );

                REQUIRE_THAT( ( grad_fd[i] - grad[i] ).norm(), WithinAbs( 0, epsilon_2 ) );
            }
        }

        // Compare Hessians
        auto hessian    = MatrixX( 3 * state->nos, 3 * state->nos );
        auto hessian_fd = MatrixX( 3 * state->nos, 3 * state->nos );
        for( const auto & interaction : hamiltonian->active_interactions() )
        {
            hessian.setZero();
            hessian_fd.setZero();

            Engine::Vectormath::Hessian(
                system_state, hessian_fd,
                [&interaction]( const auto & state, auto & gradient )
                {
                    std::fill( gradient.begin(), gradient.end(), Vector3::Zero() );
                    interaction->Gradient( state, gradient );
                } );
            interaction->Hessian( system_state, hessian );
            INFO( "Interaction: " << interaction->Name() << "\n" );
            INFO( "epsilon = " << epsilon_3 << "\n" );
            INFO( "Hessian (FD) =\n" << hessian_fd << "\n" );
            INFO( "Hessian      =\n" << hessian << "\n" );
            REQUIRE( hessian_fd.isApprox( hessian, epsilon_3 ) );
        }
    }
}

TEST_CASE( "Dipole-Dipole Interaction", "[hamiltonian]" )
{
    // Config file where only DDI is enabled
    constexpr auto input_file = "core/test/input/physics_ddi.toml";
    auto state                = std::shared_ptr<State>( State_Setup( input_file ), State_Delete );
    REQUIRE( state != nullptr );
    REQUIRE( !state->config_file.empty() );

    Configuration_Random( state.get() );
    auto & system_state = *state->active_image->state;

    auto ddi_interaction = state->active_image->hamiltonian->getInteraction<Engine::Spin::Interaction::DDI>();

    // FFT gradient and energy
    auto grad_fft = vectorfield( state->nos, Vector3::Zero() );
    ddi_interaction->Gradient( system_state, grad_fft );
    auto energy_fft = ddi_interaction->Energy( system_state );
    {
        auto grad_fft_fd = vectorfield( state->nos, Vector3::Zero() );
        Engine::Vectormath::Gradient(
            system_state, grad_fft_fd,
            [&ddi_interaction]( const auto & state ) -> scalar { return ddi_interaction->Energy( state ); } );

        INFO( "Interaction: " << ddi_interaction->Name() << "\n" );
        for( int i = 0; i < state->nos; i++ )
        {
            INFO( "i = " << i << ", epsilon = " << epsilon_3 << "\n" );
            INFO( "Gradient (FD) = " << grad_fft_fd[i].transpose() << "\n" );
            INFO( "Gradient      = " << grad_fft[i].transpose() << "\n" );
            REQUIRE_THAT( ( grad_fft_fd[i] - grad_fft[i] ).norm(), WithinAbs( 0, epsilon_3 ) );
        }
    }
    // Direct (cutoff) gradient and energy
    auto n_periodic_images = std::vector<int>{ 4, 4, 4 };
    Hamiltonian_Set_DDI( state.get(), SPIRIT_DDI_METHOD_CUTOFF, n_periodic_images.data(), -1 );
    auto grad_direct = vectorfield( state->nos, Vector3::Zero() );
    ddi_interaction->Gradient( system_state, grad_direct );
    auto energy_direct = ddi_interaction->Energy( system_state );

    // Compare gradients
    for( int i = 0; i < state->nos; i++ )
    {
        INFO( "Failed DDI-Gradient comparison at i = " << i << ", epsilon = " << epsilon_2 << "\n" );
        INFO( "Gradient (FFT)    = " << grad_fft[i].transpose() << "\n" );
        INFO( "Gradient (Direct) = " << grad_direct[i].transpose() << "\n" );
        REQUIRE_THAT( ( grad_fft[i] - grad_direct[i] ).norm(), WithinAbs( 0, epsilon_2 ) );
    }

    // Compare energies
    INFO( "Failed energy comparison test! epsilon = " << epsilon_5 );
    INFO( "Energy (Direct) = " << energy_direct << "\n" );
    INFO( "Energy (FFT)    = " << energy_fft << "\n" );
    REQUIRE_THAT( energy_fft, WithinAbs( energy_direct, epsilon_5 ) );
}

TEST_CASE( "Ensure that Hamiltonian is really just an aggregator", "[hamiltonian]" )
{
    // Hamiltonians to be tested
    for( const auto * input_file : input_files )
    {
        INFO( " Testing" << input_file );

        auto state = std::shared_ptr<State>( State_Setup( input_file ), State_Delete );
        REQUIRE( state != nullptr );
        REQUIRE( !state->config_file.empty() );

        Configuration_Random( state.get() );
        const auto & spins = *state->active_image->state;
        auto & hamiltonian = state->active_image->hamiltonian;
        auto nos           = spins.spin.size();

        if( hamiltonian->active_count() == 0 )
        {
            CAPTURE( fmt::format( " Warning: input file \"{}\" didn't specify any interaction to test.", input_file ) );
        }

        auto active_interactions = hamiltonian->active_interactions();
        auto aggregator          = [&active_interactions]( const auto init, const auto & f )
        { return std::accumulate( std::begin( active_interactions ), std::end( active_interactions ), init, f ); };

        scalar energy_hamiltonian = hamiltonian->Energy( spins );
        scalar energy_aggregated  = aggregator(
            0.0, [&spins]( const scalar v, const auto & interaction ) -> scalar
            { return v + interaction->Energy( spins ); } );

        INFO( "Hamiltonian::Energy" )
        INFO( "[total], epsilon = " << epsilon_6 << "\n" );
        INFO( "Energy (Hamiltonian) = " << energy_hamiltonian << "\n" );
        INFO( "Energy (aggregated)  = " << energy_aggregated << "\n" );
        REQUIRE_THAT( energy_hamiltonian, WithinAbs( energy_aggregated, epsilon_6 ) );

        scalarfield energy_per_spin_hamiltonian{}; // resize and clear should be handled by the hamiltonian
        hamiltonian->Energy_per_Spin( spins, energy_per_spin_hamiltonian );
        scalarfield energy_per_spin_aggregated = aggregator(
            scalarfield( nos, 0 ),
            [&spins]( const scalarfield & v, const auto & interaction ) -> scalarfield
            {
                const auto nos       = spins.spin.size();
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
            INFO( "i = " << i << ", epsilon = " << epsilon_6 << "\n" );
            INFO( "Energy (Hamiltonian)   = " << energy_per_spin_hamiltonian[i] << "\n" );
            INFO( "Energy (aggregated) = " << energy_per_spin_aggregated[i] << "\n" );
            REQUIRE_THAT( energy_per_spin_hamiltonian[i], WithinAbs( energy_per_spin_aggregated[i], epsilon_6 ) );
        }

        vectorfield gradient_hamiltonian{}; // resize and clear should be handled by the hamiltonian
        hamiltonian->Gradient( spins, gradient_hamiltonian );
        vectorfield gradient_aggregated = aggregator(
            vectorfield( nos, Vector3::Zero() ),
            [&spins]( const vectorfield & v, const auto & interaction ) -> vectorfield
            {
                auto gradient = vectorfield( spins.spin.size(), Vector3::Zero() );
                interaction->Gradient( spins, gradient );
                Engine::Vectormath::add_c_a( 1.0, v, gradient );
                return gradient;
            } );

        for( int i = 0; i < state->nos; i++ )
        {
            INFO( "Hamiltonian::Gradient" )
            INFO( "i = " << i << ", epsilon = " << epsilon_6 << "\n" );
            INFO( "Gradient (Hamiltonian)   = " << gradient_hamiltonian[i] << "\n" );
            INFO( "Gradient (aggregated) = " << gradient_aggregated[i] << "\n" );
            REQUIRE( gradient_hamiltonian[i].isApprox( gradient_aggregated[i], epsilon_6 ) );
        }

        scalar energy_combined_hamiltonian = 0;
        vectorfield gradient_combined_hamiltonian{};
        energy_combined_hamiltonian = hamiltonian->Gradient_and_Energy( spins, gradient_combined_hamiltonian );

        for( int i = 0; i < state->nos; i++ )
        {
            INFO( "Hamiltonian::Gradient_and_Energy" )
            INFO( "i = " << i << ", epsilon = " << epsilon_6 << "\n" );
            INFO( "Gradient (combined)   = " << gradient_combined_hamiltonian[i] << "\n" );
            INFO( "Gradient (aggregated) = " << gradient_aggregated[i] << "\n" );
            REQUIRE( gradient_combined_hamiltonian[i].isApprox( gradient_aggregated[i], epsilon_6 ) );
        }

        INFO( "Hamiltonian::Gradient_and_Energy" )
        INFO( "[total], epsilon = " << epsilon_6 << "\n" );
        INFO( "Energy (combined)   = " << energy_combined_hamiltonian << "\n" );
        INFO( "Energy (aggregated) = " << energy_aggregated << "\n" );
        REQUIRE_THAT( energy_combined_hamiltonian, WithinAbs( energy_aggregated, epsilon_6 ) );
    }
}
