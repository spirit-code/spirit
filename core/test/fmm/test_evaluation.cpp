#include <fmm/Box.hpp>
#include <fmm/Formulas.hpp>
#include <fmm/SimpleFMM_Defines.hpp>
#include <fmm/Tree.hpp>
#include <fmm/Utility.hpp>

#include <catch.hpp>

using SimpleFMM::Utility::multipole_idx;

TEST_CASE( "FMM: evaluation", "[fmm]" )
{
    // Build a system
    SimpleFMM::vectorfield pos;
    SimpleFMM::vectorfield spins;
    SimpleFMM::scalarfield mu_s;
    SimpleFMM::intfield pos_indices;
    SimpleFMM::intfield pos_indices1;
    SimpleFMM::intfield pos_indices2;

    int Na           = 4;
    int Nb           = 4;
    int Nc           = 1;
    int l_min        = 2;
    int l_max        = 10;
    int degree_local = l_max;

    for( int a = 0; a < Na; a++ )
    {
        for( int b = 0; b < Nb; b++ )
        {
            for( int c = 0; c < Nc; c++ )
            {
                pos.push_back( { (double)a, (double)b, (double)c } );
                spins.push_back( { 1.0, 1.0, 1.0 } );
                mu_s.push_back( 1.0 );
                pos_indices.push_back( pos_indices.size() );
                pos_indices1.push_back( pos_indices1.size() );
            }
        }
    }

    for( int a = 0; a < Na; a++ )
    {
        for( int b = 0; b < Nb; b++ )
        {
            for( int c = 0; c < Nc; c++ )
            {
                pos.push_back( { 10 + (double)a, 10 + (double)b, (double)c } );
                spins.push_back( { 1.0, 1.0, 1.0 } );
                mu_s.push_back( 1.0 );
                pos_indices.push_back( pos_indices.size() );
                pos_indices2.push_back( pos_indices2.size() + pos_indices1.size() );
            }
        }
    }

    // This box contains all positions
    SimpleFMM::Box box( pos, pos_indices, 0, l_max, degree_local );

    // This box contains the left positons
    SimpleFMM::Box box1( pos, pos_indices1, 0, l_max, degree_local );

    // This box contains the reight positions
    SimpleFMM::Box box2( pos, pos_indices2, 0, l_max, degree_local );

    INFO( box.Info_String() << "\n" << box1.Info_String() << "\n" << box2.Info_String() );

    // Ideal gradient
    SimpleFMM::vectorfield gradient( pos.size() );
    box.Evaluate_Near_Field( spins, mu_s, gradient );
    // Approximate Gradient
    SimpleFMM::vectorfield gradient_approx( pos.size() );

    Get_Multipole_Hessians( box1, l_min, l_max, 1e-3 );
    Calculate_Multipole_Moments( box1, spins, mu_s, 2, l_max );
    Get_Multipole_Hessians( box2, l_min, l_max, 1e-3 );
    Calculate_Multipole_Moments( box2, spins, mu_s, 2, l_max );
    M2L( box1, box2, l_min, l_max, degree_local );
    M2L( box2, box1, l_min, l_max, degree_local );

    SimpleFMM::Vector3 r = { 1, 1, 1 };
    INFO( box1.Info_String() << "\n" << box2.Info_String() );
    INFO(
        box2.Evaluate_Far_Field_At( box2.center + r )
        << "\n   ---\n"
        << box1.Evaluate_Directly_At( box2.center + r, spins ) << "\n   ---\n"
        << box1.Evaluate_Multipole_Expansion_At( box2.center + r ) );

    // box1.Evaluate_Near_Field(spins, gradient_approx);
    // box2.Evaluate_Near_Field(spins, gradient_approx);

    // box1.Evaluate_Far_Field(gradient_approx);
    // box2.Evaluate_Far_Field(gradient_approx);

    for( int i = 0; i < pos.size(); i++ )
    {
        INFO( "================\n" << gradient[i] << "\n   ---\n" << gradient_approx[i] << "\n================" );
    }
}