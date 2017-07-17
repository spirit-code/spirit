#include <catch.hpp>
#include <Spirit/State.h>
#include <Spirit/Configurations.h>
#include <Spirit/Parameters.h>
#include <Spirit/Geometry.h>
#include <Spirit/Simulation.h>
#include <Spirit/System.h>
#include <Spirit/Quantities.h>

TEST_CASE( "Optimizers testing", "[optimizers]" )
{
    // input file
    auto inputfile = "core/test/input/optimizers/test.cfg";

    // optimizers to be tested
    std::vector<const char *>  optimizers { "VP", "Heun", "SIB" };

    // expected values
    float energy_expected = -5139.478515625f;
    std::vector<float> magnetization_expected{ 0, 0, 0.77687f };

    // result values
    scalar energy;
    std::vector<float> magnetization{ 0, 0, 0 };

    // simulation parameters
    auto method = "LLG";

    // setup
    auto state = std::shared_ptr<State>( State_Setup( inputfile ), State_Delete );

    // calculate energy and magnetization for every optimizer
    for ( auto opt : optimizers )
    {
        // put a skyrmion in the center of the space
        Configuration_PlusZ( state.get() );
        Configuration_Skyrmion( state.get(), 5, 1, -90, false, false, false);

        // do simulation
        Simulation_PlayPause( state.get(), method, opt );

        // save energy and magnetization
        energy = System_Get_Energy( state.get() );
        Quantity_Get_Magnetization( state.get(), magnetization.data() );
            
        // log the name of the optimizer
        INFO( opt );

        // check the values of energy and magnetization
        REQUIRE( energy == Approx( energy_expected ) );
        for (int dim=0; dim<3; dim++)
            REQUIRE( magnetization[dim] == Approx( magnetization_expected[dim] ) );
    }    
}