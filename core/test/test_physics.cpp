#include <catch.hpp>
#include <Spirit/State.h>
#include <Spirit/System.h>
#include <Spirit/Simulation.h>
#include <Spirit/Configurations.h>
#include <Spirit/Hamiltonian.h>
#include <Spirit/Constants.h>
#include <Spirit/Parameters.h>
#include <data/State.hpp>
#include <Eigen/Dense>
#include <Eigen/Core>
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
    
    // Solvers to be tested
    std::vector<const char *>  solvers{ "Depondt", "Heun", "SIB" };
    
    // set up one the initial direction of the spin
    float init_direction[3] = { 1., 0., 0. };                // vec parallel to x-axis
    Configuration_Domain( state.get(), init_direction );     // set spin parallel to x-axis
    
    // assure that the initial direction is set
    auto direction = System_Get_Spin_Directions( state.get() );
    REQUIRE( direction[0] == 1 );
    REQUIRE( direction[1] == 0 );
    REQUIRE( direction[2] == 0 );
    
    // make sure that mu_s is the same as the one define in input file
    int n_cell_atoms = Geometry_Get_N_Cell_Atoms(state.get());
    std::vector<float> mu_s(n_cell_atoms, 1);
    Geometry_Get_mu_s( state.get(), mu_s.data() );
    REQUIRE( mu_s[0] == 2 );
    
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
    
    for( auto opt : solvers )
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
            
            REQUIRE( direction[2] == 0 );     // spin should stay in the xy plane
            
            // TODO: should not be scaled by mu_s
            REQUIRE( Approx(angle)  == ( (i+1) * tstep * mu_s[0] * Constants_gamma() * B_mag ) );
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
            // TODO: should not be scaled by mu_s
            projection = std::tanh( damping * (i+1) * tstep * mu_s[0] * B_mag / (2*Constants_Pi()) );
            
            direction = System_Get_Spin_Directions( state.get() );
            
            REQUIRE( Approx( projection ) == direction[2] );
        }
    }
}

TEST_CASE( "Finite Differences", "[physics]" )
{
    // Hamiltonians to be tested
    std::vector<const char *>  hamiltonians{ "core/test/input/fd_pairs.cfg" };
                                             //"core/test/input/fd_neighbours",
                                             //"core/test/input/fd_gaussian.cfg"};
    for( auto ham: hamiltonians )
    {
        INFO( " Testing " << ham );
        
        // create state
        auto state = std::shared_ptr<State>( State_Setup( ham ), State_Delete );
        
        Configuration_Random( state.get() );
        
        auto& vf = *state->active_image->spins;
        auto grad = vectorfield( state->nos );
        auto grad_fd = vectorfield( state->nos );
        
        state->active_image->hamiltonian->Gradient_FD( vf, grad_fd );
        state->active_image->hamiltonian->Gradient( vf, grad );
        
        for( int i=0; i<state->nos; i++)
            REQUIRE( grad_fd[i].isApprox( grad[i] ) );
        
        auto hessian = MatrixX( 3*state->nos, 3*state->nos );
        auto hessian_fd = MatrixX( 3*state->nos, 3*state->nos );
        
        state->active_image->hamiltonian->Hessian_FD( vf, hessian_fd );
        state->active_image->hamiltonian->Hessian( vf, hessian );
        
        REQUIRE( hessian_fd.isApprox( hessian ) );
    }
}
