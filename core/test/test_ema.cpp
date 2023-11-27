#include <Spirit/Configurations.h>
#include <Spirit/Parameters_EMA.h>
#include <Spirit/Simulation.h>
#include <Spirit/State.h>
#include <data/State.hpp>

#include <catch.hpp>

constexpr auto inputfile = "core/test/input/fd_pairs.cfg";
// constexpr auto testfile = "method_EMA_test.txt";

TEST_CASE( "Trivial", "[EMA]" )
{
    auto state = std::shared_ptr<State>( State_Setup( inputfile ), State_Delete );

    Parameters_EMA_Set_N_Modes( state.get(), 4 );
    Parameters_EMA_Set_N_Mode_Follow( state.get(), 0 );
    Parameters_EMA_Set_Frequency( state.get(), 0.02 );
    Parameters_EMA_Set_Amplitude( state.get(), 1 );
    Parameters_EMA_Set_Snapshot( state.get(), false );
    Parameters_EMA_Set_Sparse( state.get(), false );

    Configuration_PlusZ( state.get() );

    Simulation_EMA_Start( state.get(), 20 );
    const auto & spins = *state->active_image->spins;
    auto gradient      = vectorfield( state->nos );
    state->active_image->hamiltonian->Gradient( spins, gradient );
    Vector3 gradient_expected{ 6, -6, -22.8942 };

    for( int i = 0; i < 1; i++ )
    {
        INFO( "Failed EMA-Gradient comparison at i = " << i );
        INFO( "Gradient (EMA):      " << gradient[i].transpose() << "\n" );
        INFO( "Gradient (expected): " << gradient_expected.transpose() << "\n" );
        REQUIRE( gradient[i].isApprox( gradient_expected, 1e-4 ) );
    }
}
