#include <catch.hpp>
#include <Spirit/State.h>
#include <Spirit/System.h>
#include <Spirit/Simulation.h>
#include <Spirit/Configurations.h>
#include <Spirit/Hamiltonian.h>
#include <Spirit/Constants.h>
#include <Spirit/Parameters.h>
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
    
    // get the magnitude of the magnetic field ( it has only z-axis component )
    float B_mag;
    float normal[3];
    Hamiltonian_Get_Field( state.get(), &B_mag, normal );
    
    // get time step of method
    float tstep = Parameters_Get_LLG_Time_Step( state.get() );
    
    // define damping values
    scalar no_damping = 0.0;        // for measuring Larmor precession
    scalar damping    = 0.3;        // for measuring precession projection to z-axis
    
    // testing values
    float angle, projection;
    
    for( auto opt : optimizers )
    {
        // Test Larmor frequency
        
        Configuration_Domain( state.get(), init_direction );     // set spin parallel to x-axis
        Parameters_Set_LLG_Damping( state.get(), no_damping );
        
        for( int i=0; i<5; i++ )
        {
            INFO( std::string( opt ) << " failed Larmor frequency test at " << i << " iteration." );
            
            Simulation_SingleShot( state.get(), method, opt );
            
            direction = System_Get_Spin_Directions( state.get() );
            angle = std::atan2( direction[1] , direction[0] );
            
            REQUIRE( direction[2] == 0 );     // spin should always be to the xy plane
            
            // gamma must be scalled by mu_s
            REQUIRE( Approx(angle)  == ( (i+1) * tstep * mu_s * Constants_gamma() * B_mag ) );
        }
        
        // Test precession orbit projection on z-axis
        
        Configuration_Domain( state.get(), init_direction );     // set spin parallel to x-axis
        Parameters_Set_LLG_Damping( state.get(), damping );
        
        REQUIRE( direction[2] == 0 );     // initial spin position must have z=0
        
        for( int i=0; i<5; i++ )
        {
            INFO( std::string( opt ) << " failed orbit projection to z-axis test at " << 
                  i << " iteration." );
            
            Simulation_SingleShot( state.get(), method, opt );
            
            // analytical calculation of the projection in the z-axis
            projection = std::tanh( damping * (i+1) * tstep * mu_s * B_mag / (2*M_PI) );
            
            direction = System_Get_Spin_Directions( state.get() );
            
            REQUIRE( Approx( projection ) == direction[2] );
        }
    }
}