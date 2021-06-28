#include "test_util.hpp"

#include <fmm/Box.hpp>
#include <fmm/Formulas.hpp>
#include <fmm/SimpleFMM_Defines.hpp>
#include <fmm/Tree.hpp>

#include <catch.hpp>

using namespace SimpleFMM;

TEST_CASE( "FMM: M2M", "[fmm]" )
{
    // Build a system
    SimpleFMM::vectorfield pos;
    SimpleFMM::vectorfield spins;
    SimpleFMM::scalarfield mu_s;
    SimpleFMM::intfield pos_indices;

    int Na           = 4;
    int Nb           = 4;
    int Nc           = 4;
    int l_min        = 2;
    int l_max        = 6;
    int degree_local = 2;

    Testing::generate_geometry( pos, { 4, 4, 4 }, { 3, 3, 3 } );
    Testing::generate_spins( spins, mu_s, pos.size(), 1.7902 );

    // We build a tree with two levels
    auto tree    = SimpleFMM::Tree( 2, pos, 3, l_max, degree_local );
    Box & my_box = tree.Get_Box( 0 );

    // std::vector<Box> children = std::move(my_box.Divide_Evenly(3));

    // First my_box calculates the multipole moments itself
    Get_Multipole_Hessians( my_box, l_min, l_max, 1e-2 );
    SimpleFMM::Calculate_Multipole_Moments( my_box, spins, mu_s, l_min, l_max );

    // Save for later
    std::vector<Vector3c> exact_moments( my_box.multipole_moments );

    INFO( "Before upward pass\n" << my_box.Info_String( true ) );

    // Now my_box calculates the moments via M2M
    tree.Upward_Pass( spins, mu_s );

    INFO( "After upward pass\n" << my_box.Info_String( true ) );

    for( int l = l_min; l <= l_max; l++ )
    {
        for( int m = -l; m <= l; m++ )
        {
            int i = SimpleFMM::Utility::multipole_idx( l, m, l_min );

            std::stringstream ss1;
            ss1 << exact_moments[i];
            std::stringstream ss2;
            ss2 << my_box.multipole_moments[i];

            INFO( fmt::format(
                "----- {} {} -----\n"
                "----- exact moment -----\n"
                "{}\n"
                "----- M2M moment -----\n"
                "{}",
                l, m, ss1.str(), ss2.str() ) );

            REQUIRE( exact_moments[i].isApprox( my_box.multipole_moments[i], 1e-3 ) );
        }
    }
}