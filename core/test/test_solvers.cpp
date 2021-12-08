#include <Spirit/Chain.h>
#include <Spirit/Configurations.h>
#include <Spirit/Geometry.h>
#include <Spirit/Parameters_GNEB.h>
#include <Spirit/Parameters_LLG.h>
#include <Spirit/Quantities.h>
#include <Spirit/Simulation.h>
#include <Spirit/State.h>
#include <Spirit/System.h>
#include <Spirit/Transitions.h>
#include <Spirit/Version.h>
#include <string.h>
#include <catch.hpp>
#include <iostream>

TEST_CASE( "Solvers testing", "[solvers]" )
{
    // Input file
    auto inputfile = "core/test/input/solvers.cfg";

    // State
    auto state = std::shared_ptr<State>( State_Setup( inputfile ), State_Delete );

    // Reduce precision if float accuracy
    double epsilon_apprx = 1e-5;
    if( strcmp( Spirit_Scalar_Type(), "float" ) == 0 )
    {
        WARN( "Detected single precision calculation. Reducing precision requirements." );
        Parameters_LLG_Set_Convergence( state.get(), 1e-5 );
        Parameters_GNEB_Set_Convergence( state.get(), 1e-4 );
        epsilon_apprx = 5e-3;
    }

    // Solvers to be tested
    std::vector<int> solvers{ Solver_LBFGS_Atlas, Solver_LBFGS_OSO, Solver_VP_OSO,  Solver_VP,
                              Solver_Heun,        Solver_SIB,       Solver_Depondt, Solver_RungeKutta4 };

    // Expected values
    float energy_expected = -5849.69140625f;
    std::vector<float> magnetization_expected{ 0, 0, 0.79977f };

    // Result values
    scalar energy;
    std::vector<float> magnetization{ 0, 0, 0 };

    // Calculate energy and magnetization for every solvers with direct minimization
    Parameters_LLG_Set_Direct_Minimization( state.get(), true );
    for( auto solver : solvers )
    {
        // Put a skyrmion in the center of the space
        Configuration_PlusZ( state.get() );
        Configuration_Skyrmion( state.get(), 5, 1, -90, false, false, false );

        // Do simulation
        Simulation_LLG_Start( state.get(), solver );

        // Save energy and magnetization
        energy = System_Get_Energy( state.get() );
        Quantity_Get_Magnetization( state.get(), magnetization.data() );

        // Log the name of the solvers
        INFO( "LLG using " << solver << " solver (direct)" );

        // Check the values of energy and magnetization
        REQUIRE( energy == Approx( energy_expected ).epsilon( epsilon_apprx ) );
        for( int dim = 0; dim < 3; dim++ )
            REQUIRE( magnetization[dim] == Approx( magnetization_expected[dim] ).epsilon( epsilon_apprx ) );
    }

    Chain_Image_to_Clipboard( state.get() );
    int noi = 9;

    for( int i = 1; i < noi; ++i )
        Chain_Insert_Image_After( state.get() );

    // Solvers to be tested
    solvers = { Solver_LBFGS_Atlas, Solver_LBFGS_OSO, Solver_VP_OSO, Solver_VP, Solver_Heun, Solver_Depondt };

    // Expected values
    float energy_sp_expected = -5811.5244140625f;
    std::vector<float> magnetization_sp_expected{ 0, 0, 0.96657f };

    // Result values
    scalar energy_sp;
    std::vector<float> magnetization_sp{ 0, 0, 0 };

    // Calculate energy and magnetization at saddle point for every solver
    for( auto solver : solvers )
    {
        // Create a skyrmion collapse transition
        Chain_Replace_Image( state.get(), 0 );
        Chain_Jump_To_Image( state.get(), noi - 1 );
        Configuration_PlusZ( state.get() );
        Chain_Jump_To_Image( state.get(), 0 );
        Transition_Homogeneous( state.get(), 0, noi - 1 );

        // Do simulation
        Simulation_GNEB_Start( state.get(), solver, 2e4 );
        Parameters_GNEB_Set_Image_Type_Automatically( state.get() );
        Simulation_GNEB_Start( state.get(), solver );

        // Get saddle point index
        int i_max   = 1;
        float E_max = System_Get_Energy( state.get(), 0 );

        float E_temp = 0;
        for( int i = 1; i < noi - 1; ++i )
        {
            E_temp = System_Get_Energy( state.get(), i );
            if( E_temp > E_max )
            {
                i_max = i;
                E_max = E_temp;
            }
        }

        // Save energy and magnetization
        energy_sp = E_max;
        Quantity_Get_Magnetization( state.get(), magnetization_sp.data(), i_max );

        // Log the name of the solver
        INFO( "GNEB using " << solver << " solver" );

        // Check the values of energy and magnetization
        REQUIRE( energy_sp == Approx( energy_sp_expected ).epsilon( epsilon_apprx ) );
        for( int dim = 0; dim < 3; dim++ )
            REQUIRE( magnetization_sp[dim] == Approx( magnetization_sp_expected[dim] ).epsilon( epsilon_apprx ) );
    }
}