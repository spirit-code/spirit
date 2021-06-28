#include <fmm/Box.hpp>
#include <fmm/SimpleFMM_Defines.hpp>
#include <fmm/Tree.hpp>
#include <fmm/Utility.hpp>

#include <catch.hpp>

TEST_CASE( "FMM: downward pass", "[fmm]" )
{
    SimpleFMM::vectorfield pos;
    SimpleFMM::vectorfield spins;
    SimpleFMM::scalarfield mu_s;
    SimpleFMM::intfield indices;

    int Na    = 32;
    int Nb    = 1;
    int Nc    = 1;
    int l_max = 4;
    int l_min = 2;

    // Build a simple cubic grid and some spins
    for( int a = 0; a < Na; a++ )
    {
        for( int b = 0; b < Nb; b++ )
        {
            for( int c = 0; c < Nc; c++ )
            {
                pos.push_back( { (double)a, (double)b, (double)c } );
                spins.push_back( { 1.0, 1.0, 1.0 } );
                indices.push_back( indices.size() );
                mu_s.push_back( 1.0 );
            }
        }
    }

    // Build a tree
    SimpleFMM::Tree tree = SimpleFMM::Tree( 4, pos, 1, l_max );
    tree.Upward_Pass( spins, mu_s );
    tree.Downward_Pass();

    for( auto it = tree.begin_level( 0 ); it != tree.end_level( 3 ); it++ )
    {
        INFO( it->Info_String( false, false ) );
    }
}