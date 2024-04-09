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

#include "catch.hpp"

#include <cmath>
#include <string>

// Reduce required precision if float accuracy
#ifdef SPIRIT_SCALAR_TYPE_DOUBLE
[[maybe_unused]] constexpr int digits_a = 7;
[[maybe_unused]] constexpr int digits_b = 8;
[[maybe_unused]] constexpr int digits_c = 5;
#else
[[maybe_unused]] constexpr int digits_a = 6;
[[maybe_unused]] constexpr int digits_b = 4;
[[maybe_unused]] constexpr int digits_c = 1;
#endif

template<typename T>
auto within_digits( T value, int decimals_required_equal )
{
    double using_decimals = decimals_required_equal - int( std::ceil( std::log10( std::abs( value ) ) ) );
    INFO(
        "Requested " << decimals_required_equal << " decimals, meaning " << using_decimals
                     << " decimals after the floating point" );
    return Catch::Matchers::WithinAbs( value, std::pow( 10, -using_decimals ) );
}

constexpr auto inputfile = "core/test/input/solvers.cfg";

TEST_CASE( "Solvers should find Skyrmion energy minimum with direct minimization", "[solvers]" )
{
    std::vector<int> solvers{
        Solver_LBFGS_Atlas, Solver_LBFGS_OSO, Solver_VP_OSO,  Solver_VP,
        Solver_Heun,        Solver_SIB,       Solver_Depondt, Solver_RungeKutta4,
    };
    // Expected values
    scalar energy_expected = -5849.69140625f;
    std::vector<scalar> magnetization_expected{ 0, 0, 2.0 * 0.79977 };

    auto state = std::shared_ptr<State>( State_Setup( inputfile ), State_Delete );

    // Reduce convergence threshold if float accuracy
    if( std::string( Spirit_Scalar_Type() ) == "float" )
    {
        WARN( "Detected single precision calculation. Reducing LLG convergence threshold to 1e-5" );
        Parameters_LLG_Set_Convergence( state.get(), 1e-5 );
    }

    // Calculate energy and magnetization for every solvers with direct minimization
    Parameters_LLG_Set_Direct_Minimization( state.get(), true );
    for( auto solver : solvers )
    {
        INFO( "Direct minimisation using " << solver << " solver" );

        // Put a skyrmion in the center of the space
        Configuration_PlusZ( state.get() );
        Configuration_Skyrmion( state.get(), 5, 1, -90, false, false, false );

        // Perform simulation until convergence
        Simulation_LLG_Start( state.get(), solver );

        // Check the values of energy and magnetization
        scalar energy = System_Get_Energy( state.get() );
        std::vector<scalar> magnetization{ 0, 0, 0 };
        Quantity_Get_Magnetization( state.get(), magnetization.data() );
        REQUIRE_THAT( energy, within_digits( energy_expected, digits_b ) );
        for( int dim = 0; dim < 3; dim++ )
            REQUIRE_THAT( magnetization[dim], within_digits( magnetization_expected[dim], digits_c ) );
    }
}

TEST_CASE( "Solvers should find Skyrmion collapse barrier with GNEB method", "[solvers]" )
{
    constexpr int NOI = 9;
    // Solvers to be tested
    std::vector<int> solvers = {
        Solver_LBFGS_Atlas, Solver_LBFGS_OSO, Solver_VP_OSO, Solver_VP, Solver_Heun, Solver_Depondt,
    };
    // Expected values
    scalar energy_sp_expected = -5811.5244140625;
    std::vector<scalar> magnetization_sp_expected{ 0, 0, 2.0 * 0.96657 };

    auto state = std::shared_ptr<State>( State_Setup( inputfile ), State_Delete );

    // Reduce convergence threshold if float accuracy
    if( std::string( Spirit_Scalar_Type() ) == "float" )
    {
        WARN( "Detected single precision calculation. Reducing LLG convergence threshold to 1e-5 and GNEB to 1e-4" );
        Parameters_LLG_Set_Convergence( state.get(), 1e-5 );
        Parameters_GNEB_Set_Convergence( state.get(), 1e-4 );
    }

    Configuration_PlusZ( state.get() );
    Chain_Image_to_Clipboard( state.get() );
    Chain_Set_Length( state.get(), NOI );

    // Calculate energy and magnetization at saddle point for every solver
    for( auto solver : solvers )
    {
        INFO( "GNEB using " << solver << " solver" );

        // Put a skyrmion in the center of the space
        Chain_Jump_To_Image( state.get(), 0 );
        Configuration_PlusZ( state.get() );
        Configuration_Skyrmion( state.get(), 5, 1, -90, false, false, false );

        // Perform simulation until convergence
        Simulation_LLG_Start( state.get(), solver );

        // Create a skyrmion collapse transition
        Chain_Jump_To_Image( state.get(), NOI - 1 );
        Configuration_PlusZ( state.get() );
        Chain_Jump_To_Image( state.get(), 0 );
        Transition_Homogeneous( state.get(), 0, NOI - 1 );

        // Perform simulation until convergence
        Simulation_GNEB_Start( state.get(), solver, 20'000 );
        Parameters_GNEB_Set_Image_Type_Automatically( state.get() );
        Simulation_GNEB_Start( state.get(), solver );

        // Get saddle point index and energy
        int idx_sp       = 1;
        scalar energy_sp = System_Get_Energy( state.get(), 0 );
        for( int idx = 1; idx < NOI - 1; ++idx )
        {
            scalar E_temp = System_Get_Energy( state.get(), idx );
            if( E_temp > energy_sp )
            {
                idx_sp    = idx;
                energy_sp = E_temp;
            }
        }

        // Check the values of energy and magnetization
        std::vector<scalar> magnetization_sp{ 0, 0, 0 };
        Quantity_Get_Magnetization( state.get(), magnetization_sp.data(), idx_sp );
        REQUIRE_THAT( energy_sp, within_digits( energy_sp_expected, digits_a ) );
        for( int dim = 0; dim < 3; dim++ )
            REQUIRE_THAT( magnetization_sp[dim], within_digits( magnetization_sp_expected[dim], 6 ) );
    }
}
