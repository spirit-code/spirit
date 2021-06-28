#include "test_util.hpp"

#include <fmm/Box.hpp>
#include <fmm/Formulas.hpp>
#include <fmm/SimpleFMM_Defines.hpp>
#include <fmm/Tree.hpp>
#include <fmm/Utility.hpp>

#include <catch.hpp>

using namespace SimpleFMM;

TEST_CASE( "FMM: L2L", "[fmm]" )
{
    SimpleFMM::vectorfield pos1;
    SimpleFMM::vectorfield spins1;
    SimpleFMM::scalarfield mu_s1;

    SimpleFMM::vectorfield pos2;
    SimpleFMM::vectorfield spins2;
    SimpleFMM::scalarfield mu_s2;

    int l_min        = 2;
    int l_max        = 6;
    int degree_local = 6;

    // Build a geometry of two boxes
    // We will use box1 to create a local expansion around the center of box2
    Testing::generate_geometry( pos1, { 4, 4, 4 }, { 0, 0, 0 } );
    Testing::generate_spins( spins1, mu_s1, pos1.size(), 1.17, { 0, 0, 1 } );
    SimpleFMM::Box box1( pos1, 0, l_max, degree_local );

    // We also need the children of box2
    Testing::generate_geometry( pos2, { 4, 4, 4 }, { 10, 10, 10 } );
    Testing::generate_spins( spins2, mu_s2, pos2.size(), 1.8, { 0, 0, 1 } );
    SimpleFMM::Box box2( pos2, 0, l_max, degree_local );
    std::vector<Box> children2 = box2.Divide_Evenly();

    // Calculate the Multipole Expansion of box1
    Get_Multipole_Hessians( box1, l_min, l_max, 1e-1 );
    Calculate_Multipole_Moments( box1, spins1, mu_s1, l_min, l_max );

    // Transform the multipole expansion of box1 into a local expansion around the center of box2
    box2.Clear_Moments();
    Cache_M2L_values( box2, box1, l_max, degree_local, l_min );
    M2L( box2, box1, l_min, l_max, degree_local ); // We now have a local expansion around the center of box2

    // Perform L2L for every child of box2
    for( auto & child : children2 )
    {
        child.Clear_Moments();
        Cache_L2L_values( box2, child, degree_local );
        Add_Local_Moments( box2, child, degree_local );
        Build_Far_Field_Cache( child, degree_local );
    }

    // Evaluate the far field at each position of the parent box using the original local expansion
    vectorfield gradient_farfield_parent( box2.n_spins, { 0, 0, 0 } );
    Build_Far_Field_Cache( box2, degree_local );
    Evaluate_Far_Field( box2, gradient_farfield_parent, mu_s2, degree_local );

    // Evaluate the far field at each position of the child boxes using the local expansion obtained by L2L and compare
    for( auto & child : children2 )
    {
        vectorfield gradient_farfield_child(
            box2.n_spins, { 0, 0, 0 } ); // Note that this vectorfield needs to have the length of box2.n_spins too
        Evaluate_Far_Field( child, gradient_farfield_child, mu_s2, degree_local );

        for( int i = 0; i < child.n_spins; i++ )
        {
            Vector3 & grad_1 = gradient_farfield_child[child.pos_indices[i]];
            Vector3 & grad_2 = gradient_farfield_parent[child.pos_indices[i]];

            INFO(
                "grad_1 = " << Testing::vector3_to_string( grad_1 )
                            << ", grad_2 = " << Testing::vector3_to_string( grad_2 ) );
            REQUIRE( grad_1.isApprox( grad_2, 1e-1 ) );
        }
    }
}