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
                                          "SIB2",   // TODO: remove in the future 
                                          "VP"
                                          //"NCG",
                                          //"Heun"
                                        };   // optimizers to be tested
  
  // expected values
  float Energy = -5094.5556640625f;
  std::vector<float> Magnetization{ 0, 0, 0.644647f };

  int index;    // index of optimizer in optimizers list  
  scalar energy;    // enery output
  std::vector<float> magnetization{ 0, 0, 0 };   // magnetization output
  
  // simulation parameters
  const char * method = "LLG";
  int n_iter;
  int n_iter_log;
  
  // skyrmion characteristics
  float radius  = 5;
  int order     = 1;
  float phase   = 0;
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
    
    // put a skyrmion with phase 0 in the center of the space
    Geometry_Get_Center( state.get(), position );
    Configuration_Skyrmion( state.get(), radius, order, phase, updown, achiral, rl, position,
                            r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted );
    
    // do simulation
    Parameters_Get_LLG_N_Iterations( state.get(), &n_iter, &n_iter_log );
    Simulation_PlayPause( state.get(), method, opt, n_iter, n_iter_log );
    
    // save energy and magnetization
    energy = System_Get_Energy( state.get() );
    Quantity_Get_Magnetization( state.get(), magnetization.data() );
        
    // log the name of the optimizer
    INFO( opt );

    // check the values of energy and magnetization
    REQUIRE( energy == Approx( Energy ) );
    for (int j=0; j<3; j++)
      REQUIRE( magnetization[j] == Approx( Magnetization[j] ) );
  }    
}