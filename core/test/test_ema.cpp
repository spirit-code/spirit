#include <catch.hpp>
#include <Spirit/State.h>
#include <Spirit/System.h>
#include <Spirit/Simulation.h>
#include <Spirit/Configurations.h>
#include <Spirit/Hamiltonian.h>
#include <Spirit/Constants.h>
#include <Spirit/IO.h>
#include <data/State.hpp>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <iostream>
#include <iomanip>
#include <sstream>

auto inputfile = "core/test/input/fd_pairs.cfg";
// auto testfile = "method_EMA_test.txt";

TEST_CASE("Trivial", "[EMA]")
{
    // create State
    auto state = std::shared_ptr<State>( State_Setup( inputfile ), State_Delete );
    
    Configuration_PlusZ( state.get() );
    
    // NOTE: the choise of a solver does not affect the simulation for EMA. We have to provide a
    // different API function for Methods (like MC and EMA) and a different for methods with solvers
    // (Heun, Depondt etc). Or even an EgeinAnalysis might be appropriate.
    
    //Simulation_SingleShot( state.get(), method, "Heun" );
    Simulation_EMA_Start( state.get(), 20 );
    //IO_Image_Write( state.get(), testfile );
    
    // Configuration_MinusZ( state.get() );
    // IO_Image_Write( state.get(), testfile );
}