#include <fmm/Box.hpp>
#include <fmm/SimpleFMM_Defines.hpp>
#include <fmm/Tree.hpp>
#include <fmm/Utility.hpp>
#include <fmm/Formulas.hpp>

#include <catch.hpp>

#include <iomanip>
#include <iostream>

TEST_CASE( "FMM", "[upwardpass]" )
{
    SimpleFMM::vectorfield pos;
    SimpleFMM::vectorfield spins;
    SimpleFMM::scalarfield mu_s;
    SimpleFMM::intfield indices;

    int Na    = 16;
    int Nb    = 1;
    int Nc    = 1;
    int l_max = 4;
    int l_min = 2;
    int degree_local = 4;

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
    SimpleFMM::Tree tree = SimpleFMM::Tree( 3, pos, 1, l_max );
    tree.Upward_Pass( spins, mu_s );
    for( auto it = tree.begin_level( 0 ); it != tree.end_level( 2 ); it++ )
    {
        it->Print_Info();
    }

    // Build one root box
    SimpleFMM::Box box = SimpleFMM::Box( pos, indices, 0, l_max, degree_local );
    Get_Multipole_Hessians(box, l_min, l_max );
    Calculate_Multipole_Moments(box, spins, mu_s, l_min, l_max );

    scalar epsilon = 1e-1;
    for( int l = l_min; l <= l_max; l++ )
    {
        for( int m = -l; m <= l; m++ )
        {
            // std::cout << "===tree===" << std::endl;
            // std::cout << "l = " << l << ", m = " << m << std::endl;
            // std::cout << tree.Get_Box(0).multipole_moments[SimpleFMM::Utility::multipole_idx(l,m)] << std::endl;
            // std::cout << "   --  " << std::endl;
            // std::cout << box.multipole_moments[SimpleFMM::Utility::multipole_idx(l,m)] << std::endl;
            // std::cout << "===box===" << std::endl;
            auto & m_tree = tree.Get_Box( 0 ).multipole_moments[SimpleFMM::Utility::multipole_idx( l, m, l_min )];
            auto & m_box  = box.multipole_moments[SimpleFMM::Utility::multipole_idx( l, m, l_min )];
            if( ( m_tree - m_box ).norm() > epsilon )
            {
                std::cout << "Fail at (l, m) = " << l << ", " << m << std::endl;
                std::cout << m_tree << std::endl;
                std::cout << m_box << std::endl;
                std::cout << ( m_tree - m_box ).norm() << std::endl;
            }
        }
    }
}