#include <catch.hpp>
#include <Spirit/State.h>
#include <Spirit/System.h>
#include <Spirit/Simulation.h>
#include <Spirit/Configurations.h>
#include <Spirit/Hamiltonian.h>
#include <Spirit/Constants.h>
#include <iostream>
#include <iomanip>
#include <sstream>

TEST_CASE( "Larmor Precession","[physics]" )
{
    // input file
    auto inputfile = "core/test/input/physics_larmor.cfg";
    
    // create State
    auto state = std::shared_ptr<State>( State_Setup( inputfile ), State_Delete );
    

    
    // choose method
    auto method = "LLG";
    
    // Optimizers to be tested
    std::vector<const char *>  optimizers { "Depondt", "Heun", "SIB"  };
    
    // set up one the initial direction of the spin
    float init_direction[3] = { 1., 0., 0. };                // vec parallel to x-axis
    Configuration_Domain( state.get(), init_direction );     // set spin parallel to x-axis
    
    // assure that the initial direction is set
    auto direction = System_Get_Spin_Directions( state.get() );
    REQUIRE( direction[0] == 1 );
    REQUIRE( direction[1] == 0 );
    REQUIRE( direction[2] == 0 );
    
    // make sure that mu_s is the same as the one define in input file
    float mu_s;
    Hamiltonian_Get_mu_s( state.get(), &mu_s );
    REQUIRE( mu_s == 2 );
    
    //std::vec<float> angle(20);
    float angle;
    
    for( auto opt : optimizers )
    {
        Configuration_Domain( state.get(), init_direction );     // set spin parallel to x-axis
        
        for( int i=0; i<7; i++ )
        {
            Simulation_SingleShot( state.get(), method, opt );
            
            direction = System_Get_Spin_Directions( state.get() );
            angle = std::atan2( direction[1] , direction[0] );
            
            INFO( std::string( opt ) << " failed Larmor frequency test at " << i << " iteration." );
            
            REQUIRE( direction[2] == 0 );     // spin should always be to the xy plane
            
            // gamma must be scalled by mu_s
            REQUIRE( Approx(angle)  == ( (i+1) * mu_s * Constants_gamma() ) );
        }        
    }
}