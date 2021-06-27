#include "test_util.hpp"

#include <fmm/Box.hpp>
#include <fmm/Formulas.hpp>
#include <fmm/SimpleFMM_Defines.hpp>
#include <fmm/Tree.hpp>
#include <fmm/Utility.hpp>

#include <catch.hpp>

#include <iomanip>
#include <iostream>

using namespace SimpleFMM;
using namespace Testing;

TEST_CASE( "FMM", "[M2L]" )
{
    SimpleFMM::vectorfield pos1;
    SimpleFMM::vectorfield spins1;
    SimpleFMM::scalarfield mu_s1;

    SimpleFMM::vectorfield pos2;
    SimpleFMM::vectorfield spins2;
    SimpleFMM::scalarfield mu_s2;

    int l_min        = 2;
    int l_max        = 6;
    int degree_local = 2;

    // Build a geometry of two boxes
    Testing::generate_geometry( pos1, { 4, 4, 4 }, { 0, 0, 0 } );
    Testing::generate_spins( spins1, mu_s1, pos1.size(), 1.17 );
    SimpleFMM::Box box1( pos1, 0, l_max, degree_local );

    Testing::generate_geometry( pos2, { 4, 4, 4 }, { 10, 10, 10 } );
    Testing::generate_spins( spins2, mu_s2, pos2.size(), 1.8 );
    SimpleFMM::Box box2( pos2, 0, l_max, degree_local );

    // We also construct a vector to hold the gradient due to spins in box1 evaluated at the positions in box2
    vectorfield gradient_box2_farfield( box2.n_spins, { 0, 0, 0 } );

    // Calculate the Multipole Expansion of box1
    Get_Multipole_Hessians( box1, l_min, l_max, 1e-1 );
    Calculate_Multipole_Moments( box1, spins1, mu_s1, l_min, l_max );

    // Transform the multipole expansion of box1 into a local expansion around the center of box2
    Cache_M2L_values( box2, box1, l_max, degree_local, l_min );
    Build_Far_Field_Cache( box2, degree_local );
    M2L( box2, box1, l_min, l_max, degree_local );

    // Evaluate the far field of box1 inside box2
    Evaluate_Far_Field( box2, gradient_box2_farfield, mu_s2, degree_local );

    for( int i = 0; i < box2.n_spins; i++ )
    {
        Vector3 position           = box2.pos[box2.pos_indices[i]];
        Vector3 gradient_direct    = calculate_gradient_directly( position, pos1, spins1, mu_s1 );
        Vector3 gradient_multipole = SimpleFMM::Evaluate_Multipole_Expansion_At( position, box1, 2, l_max );

        std::cout << "-------------------\n";
        Testing::print_vector( position );
        Testing::print_vector( gradient_direct );
        Testing::print_vector( gradient_multipole );
        std::cout << "-------------------\n";

        REQUIRE( gradient_direct.isApprox( gradient_multipole, 1e-2 ) );
    }
}