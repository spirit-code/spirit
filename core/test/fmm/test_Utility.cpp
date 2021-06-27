#include <fmm/SimpleFMM_Defines.hpp>
#include <fmm/Spherical_Harmonics.hpp>
#include <fmm/Utility.hpp>

#include <catch.hpp>

#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>

using namespace SimpleFMM;
using SimpleFMM::Spherical_Harmonics::PI;

TEST_CASE( "FMM", "[Utility]" )
{
    int l_min = 4;
    int l_max = 10;

    // n_moments and multipole_idx
    int counter = 0;
    for( int l = l_min; l <= l_max; l++ )
    {
        for( int m = -l; m <= l; m++ )
        {
            REQUIRE( Utility::multipole_idx( l, m, l_min ) == counter++ );
        }
    }
    REQUIRE( Utility::n_moments( l_max, l_min ) == counter );

    // n_moments_p and multipole_idx_p
    counter = 0;
    for( int l = l_min; l <= l_max; l++ )
    {
        for( int m = 0; m <= l; m++ )
        {
            REQUIRE( Utility::multipole_idx_p( l, m, l_min ) == counter++ );
        }
    }
    REQUIRE( Utility::n_moments_p( l_max, l_min ) == counter );

    // i_power and minus_one_power
    for( int n = 0; n < 20; n++ )
    {
        REQUIRE( Utility::minus_one_power( n ) == std::pow( -1, n ) );
        REQUIRE( std::abs( Utility::i_power( n ) - std::pow( std::complex<scalar>( 0, 1 ), n ) ) < 1e-10 );
    }

    // factorial
    scalar a_fac = 1.0;
    for( int a = 1; a <= 4; a++ )
    {
        a_fac *= a;
        scalar b_fac = 1.0;
        for( int b = 1; b <= 4; b++ )
        {
            b_fac *= b;
            REQUIRE( Utility::factorial( a, b ) == Approx( a_fac / b_fac ) );
        }
    }

    // Coordinate transformations
    for( int i = -3; i <= 3; i++ )
    {
        for( int j = -3; j <= 3; j++ )
        {
            for( int k = -3; k <= 3; k++ )
            {
                Vector3 pos_cartesian = { i, j, k };
                Vector3 pos_spherical;
                Vector3 pos_cartesian2;
                Vector3 pos_spherical2;

                Utility::get_spherical( pos_cartesian, pos_spherical );
                Utility::get_cartesian( pos_spherical, pos_cartesian2 );
                Utility::get_spherical( pos_cartesian2, pos_spherical2 );

                // std::cout << "--" << std::endl;
                // Testing::print_vector(pos_cartesian);
                // Testing::print_vector(pos_spherical);
                // Testing::print_vector(pos_cartesian2);
                // Testing::print_vector(pos_spherical2);
                // std::cout << "--" << std::endl;

                REQUIRE( pos_cartesian.isApprox( pos_cartesian2, 1e-4 ) );
                REQUIRE( pos_spherical.isApprox( pos_spherical2, 1e-4 ) );

                if( i == 0 && j > 0 )
                {
                    REQUIRE( pos_spherical[1] == Approx( PI / 2 ) );
                }
                if( i == 0 && j < 0 )
                {
                    REQUIRE( pos_spherical[1] == Approx( 3 * PI / 2 ) );
                }

                if( j == 0 && i > 0 )
                {
                    REQUIRE( pos_spherical[1] == Approx( 0 ) );
                }
                if( j == 0 && i < 0 )
                {
                    REQUIRE( pos_spherical[1] == Approx( PI ) );
                }

                if( k == 0 && ( i != 0 | j != 0 ) )
                {
                    REQUIRE( pos_spherical[2] == Approx( PI / 2 ) );
                }

                REQUIRE( pos_spherical[0] * pos_spherical[0] == Approx( i * i + j * j + k * k ) );
            }
        }
    }
}