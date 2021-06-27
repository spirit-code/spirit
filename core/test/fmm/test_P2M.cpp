#include "test_util.hpp"

#include <fmm/Box.hpp>
#include <fmm/Formulas.hpp>
#include <fmm/SimpleFMM_Defines.hpp>
#include <fmm/Tree.hpp>
#include <fmm/Utility.hpp>

#include <catch.hpp>

#include <iomanip>
#include <iostream>

using namespace Testing;

TEST_CASE( "FMM", "[P2M]" )
{
    SimpleFMM::vectorfield pos;
    SimpleFMM::vectorfield spins;
    SimpleFMM::scalarfield mu_s;
    int Na = 4, Nb = 4, Nc = 4;
    int l_max = 5;

    // Generate a lattice of 4x4x4 random but normalized spins with lower left corner at (2,2,2) mu_s is set to 1.34
    generate_geometry( pos, { Na, Nb, Nc }, { 0, 0, 0 } );
    generate_spins( spins, mu_s, Na * Nb * Nc, 1.34, { 0, 0, 0 } );

    Vector3 test_position = { 13, 9, 17 };

    // Calculate the gradient at the test position directly
    Vector3 gradient_direct = calculate_gradient_directly( test_position, pos, spins, mu_s );

    // We create a box around the particles
    SimpleFMM::Box my_box( pos, 0, l_max, l_max );

    SimpleFMM::Get_Multipole_Hessians( my_box, 2, l_max, 1e-1 );
    SimpleFMM::Calculate_Multipole_Moments( my_box, spins, mu_s, 2, l_max );
    Vector3 gradient_multipole = SimpleFMM::Evaluate_Multipole_Expansion_At( test_position, my_box, 2, l_max );

    // my_box.Print_Info(true);

    std::cout << "gradient_direct\n  ";
    print_vector( gradient_direct );
    std::cout << "gradient_direct_multipole\n  ";
    print_vector( gradient_multipole );

    REQUIRE( gradient_direct.isApprox( gradient_multipole, 1e-1 ) );
}