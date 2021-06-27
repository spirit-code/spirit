#include <fmm/Box.hpp>
#include <fmm/SimpleFMM_Defines.hpp>
#include <fmm/Tree.hpp>

#include <catch.hpp>

#include <iomanip>
#include <iostream>

TEST_CASE( "FMM", "[Octree building]" )
{
    SimpleFMM::vectorfield pos;
    SimpleFMM::vectorfield spins;
    SimpleFMM::scalarfield mu_s;

    int Na = 10;
    int Nb = 10;
    int Nc = 1;

    // Build a simple cubic grid and some spins
    for( int a = 0; a < Na; a++ )
    {
        for( int b = 0; b < Nb; b++ )
        {
            for( int c = 0; c < Nc; c++ )
            {
                pos.push_back( { (double)a, (double)b, (double)c } );
                spins.push_back( { 0.0, 0.0, 1.0 } );
                mu_s.push_back( 1.0 );
            }
        }
    }

    // Build a tree
    SimpleFMM::Tree tree = SimpleFMM::Tree( 3, pos, 2, 4, 2 );

    SimpleFMM::vectorfield gradient( spins.size(), { 0, 0, 0 } );
    SimpleFMM::vectorfield gradient_direct( spins.size(), { 0, 0, 0 } );

    for( auto it = tree.begin_level( 0 ); it != tree.end_level( 2 ); it++ )
    {
        it->Print_Info( false, false );
    }

    tree.Upward_Pass( spins, mu_s );
    tree.Downward_Pass();

    for( auto it = tree.begin_level( 0 ); it != tree.end_level( 2 ); it++ )
    {
        it->Print_Info();
    }

    tree.Evaluation( spins, mu_s, gradient );
    tree.Direct_Evaluation( spins, mu_s, gradient_direct );

    std::cout << std::fixed;
    std::cout << std::setprecision( 10 );

    for( auto i = 0; i < gradient.size(); i++ )
    {
        std::cout << "=================" << std::endl;
        std::cout << " multipole " << std::endl;
        std::cout << gradient[i] << std::endl;
        std::cout << "       --        " << std::endl;
        std::cout << " direct " << std::endl;
        std::cout << gradient_direct[i] << std::endl;
        std::cout << "=================" << std::endl;
    }
}