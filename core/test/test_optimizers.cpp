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
  // input File
  const char * inputfile = "core/test/input/optimizers/test.cfg";

  // optimizers and outputs
  std::vector<const char *>  optimizers { "SIB", 
                                          "SIB2", 
                                          "VP",
                                          "Heun" };   // optimizers to be tested
  int index;    // index of optimizer in optimizers list  
  std::vector<float>        energy( optimizers.size(), .0 );      // energy output
  std::vector<float [3]>    magnetization( optimizers.size() );   // magnetization output
  
  // simulation parameters
  const char * method = "LLG";
  int n_iter;
  int n_iter_log;
  
  // skyrmion characteristics
  float radius  = 5;
  int order     = 1;
  float phase   = -90;
  bool updown   = false,
       achiral  = false, 
       rl       = false,
       inverted = false;
  
  // skyrmion filters (default)
  float position[3]{ 0, 0, 0 },               // this must be set to center for the test
        r_cut_rectangular[3]{ -1, -1, -1 },
        r_cut_cylindrical = -1,
        r_cut_spherical   = -1;

  // calculate energy and magnetization fro every optimizer
  for ( auto opt : optimizers )
  {
    // setup
    index = find( optimizers.begin(), optimizers.end(), opt ) - optimizers.begin();
    auto state = std::shared_ptr<State>( State_Setup( inputfile ), State_Delete );
    
    // put a skyrmion with phase -90 in the center of the space
    Geometry_Get_Center( state.get(), position );
    Configuration_Skyrmion( state.get(), radius, order, phase, updown, achiral, rl, position,
                            r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted );
    
    // do simulation
    Parameters_Get_LLG_N_Iterations( state.get(), &n_iter, &n_iter_log );
    Simulation_PlayPause( state.get(), method, opt, n_iter, n_iter_log );
    
    // save energy and magnetization
    energy[ index ] = System_Get_Energy( state.get() );
    Quantity_Get_Magnetization( state.get(), magnetization[ index ] );
  
    INFO( opt );    // log the optimizer's name in case of failure
    REQUIRE( energy[ index ] != .0 );                       // energy must be different than 0
    REQUIRE( magnetization[ index ][0] == Approx( .0 ) );   // x-axis magnetization approx 0
    REQUIRE( magnetization[ index ][1] == Approx( .0 ) );   // y-axis magnetization approx 0
  }
  
  // check results between different optimizers
  for (int i=0; i<optimizers.size(); i++)
  {
    for (int k=i+1; k<optimizers.size(); k++ )
    {
      // log the name of the optimizers under comparison
      std::string opt1( optimizers[i] );
      std::string opt2( optimizers[k] );
      INFO( "While comparing " + opt1 + " - " + opt2 );
      
      // compare energy and magnetization along z-axis
      REQUIRE( energy[i] == energy[k] );                        // equal energy
      REQUIRE( magnetization[i][2] == magnetization[k][2] );    // equal magnetization
    }
  }
}