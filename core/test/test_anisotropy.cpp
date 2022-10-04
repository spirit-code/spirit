#include <Spirit/Configurations.h>
#include <Spirit/Constants.h>
#include <Spirit/Geometry.h>
#include <Spirit/Hamiltonian.h>
#include <Spirit/Parameters_LLG.h>
#include <Spirit/Simulation.h>
#include <Spirit/State.h>
#include <Spirit/System.h>
#include <Spirit/Version.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <catch.hpp>
#include <data/State.hpp>

TEST_CASE( "Anisotropy", "[anisotropy]" )
{

    double epsilon_apprx = 1e-11;
    if( strcmp( Spirit_Scalar_Type(), "float" ) == 0 )
    {
        WARN( "Detected single precision calculation. Reducing precision requirements." );
        epsilon_apprx = 1e-4;
    }

    auto state = std::shared_ptr<State>( State_Setup(), State_Delete );

    // Set the uniaxial anisotropy
    float init_magnitude = 0.1;
    float init_normal[3] = { 0., 0., 1. };
    Hamiltonian_Set_Anisotropy( state.get(), init_magnitude, init_normal );

    // Get the uniaxial anisotropy
    float magnitude;
    float normal[3];
    Hamiltonian_Get_Anisotropy( state.get(), &magnitude, normal );

    REQUIRE( Approx( magnitude ) == init_magnitude );
    REQUIRE( Approx( normal[0] ) == 0 );
    REQUIRE( Approx( normal[1] ) == 0 );
    REQUIRE( Approx( normal[2] ) == 1 );

    // Test the uniaxial anisotropy energy
    vectorfield spins( state->nos );

    for( int ispin = 0; ispin < state->nos; ++ispin )
        spins[ispin] = { 1.0, 0.0, 0.0 };

    float energy1, energy2;
    energy1 = state->active_image->hamiltonian->Energy( spins );

    for( int ispin = 0; ispin < state->nos; ++ispin )
        spins[ispin] = { 0.0, 0.0, 1.0 };

    energy2 = state->active_image->hamiltonian->Energy( spins );

    REQUIRE( Approx( energy1 - energy2 ) == init_magnitude * state->nos );

    for( int ispin = 0; ispin < state->nos; ++ispin )
        spins[ispin] = { sqrt( 2 ) / 2, sqrt( 2 ) / 2, 0.0 };

    energy2 = state->active_image->hamiltonian->Energy( spins );

    REQUIRE( Approx( energy1 - energy2 ) == 0.0 );

    for( int ispin = 0; ispin < state->nos; ++ispin )
        spins[ispin] = { 0.0, 0.0, 1.0 };

    // Test the uniaxial anisotropy gradient
    auto grad = vectorfield( state->nos );

    Vector3 grad_test = { 0.0, 0.0, -2.0 * init_magnitude };
    state->active_image->hamiltonian->Gradient( spins, grad );

    for( int ispin = 0; ispin < state->nos; ++ispin )
        REQUIRE( grad[ispin].isApprox( grad_test, epsilon_apprx ) );

    // Test the cubic anisotropy energy

    // Set uniaxial anisotropy to zero
    init_magnitude = 0.0;
    Hamiltonian_Set_Anisotropy( state.get(), init_magnitude, init_normal );

    // Set the cubic anisotropy
    float init_magnitude4 = 0.2;
    Hamiltonian_Set_Cubic_Anisotropy( state.get(), init_magnitude4 );

    // Get the cubic anisotropy
    float magnitude4;
    Hamiltonian_Get_Cubic_Anisotropy( state.get(), &magnitude4 );

    REQUIRE( Approx( magnitude4 ) == init_magnitude4 );

    for( int ispin = 0; ispin < state->nos; ++ispin )
        spins[ispin] = { 1.0, 0.0, 0.0 };

    energy1 = state->active_image->hamiltonian->Energy( spins );

    for( int ispin = 0; ispin < state->nos; ++ispin )
        spins[ispin] = { sqrt( 2 ) / 2, sqrt( 2 ) / 2, 0.0 };

    energy2 = state->active_image->hamiltonian->Energy( spins );

    REQUIRE( Approx( energy1 - energy2 ) == -init_magnitude4 / 4 * state->nos );

    for( int ispin = 0; ispin < state->nos; ++ispin )
        spins[ispin] = { 0.0, 1.0, 0.0 };

    energy1 = state->active_image->hamiltonian->Energy( spins );

    REQUIRE( Approx( energy1 - energy2 ) == -init_magnitude4 / 4 * state->nos );

    for( int ispin = 0; ispin < state->nos; ++ispin )
        spins[ispin] = { 0.0, 0.0, 1.0 };

    energy2 = state->active_image->hamiltonian->Energy( spins );

    REQUIRE( Approx( energy1 - energy2 ) == 0.0 );

    // Test the cubic anisotropy gradient
    state->active_image->hamiltonian->Gradient( spins, grad );

    grad_test = { 0.0, 0.0, -2.0 * init_magnitude4 };

    for( int ispin = 0; ispin < state->nos; ++ispin )
        REQUIRE( grad[ispin].isApprox( grad_test, epsilon_apprx ) );

    // Test the Gradient_and_Energy function for both uniaxial and cubic anisotropy
    init_magnitude = 0.1;
    Hamiltonian_Set_Anisotropy( state.get(), init_magnitude, init_normal );

    for( int ispin = 0; ispin < state->nos; ++ispin )
        spins[ispin] = { 0.0, 0.0, 1.0 };

    scalar energy3;

    state->active_image->hamiltonian->Gradient_and_Energy( spins, grad, energy3 );

    energy1 = state->active_image->hamiltonian->Energy( spins );
    REQUIRE( Approx( energy3 ) == energy1 );

    auto grad2 = vectorfield( state->nos );

    state->active_image->hamiltonian->Gradient( spins, grad2 );

    for( int ispin = 0; ispin < state->nos; ++ispin )
        REQUIRE( grad[ispin].isApprox( grad2[ispin], epsilon_apprx ) );
}
