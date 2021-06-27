#include <fmm/SimpleFMM_Defines.hpp>
#include <fmm/Spherical_Harmonics.hpp>
#include <fmm/Utility.hpp>

#include <catch.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>

using namespace SimpleFMM;
using namespace SimpleFMM::Spherical_Harmonics;
using SimpleFMM::Utility::get_spherical;

TEST_CASE( "FMM", "[Spherical harmonics]" )
{
    scalar r     = 7.2;
    scalar theta = 1.2;
    scalar phi   = 2.2;

    int l_max = 12;
    int l_min = 0;

    auto epsilon = 1e-8;

    // Test basic symmetries
    for( int l = 0; l <= l_max; l++ )
    {
        for( int m = -l; m <= l; m++ )
        {
            auto Y1 = Spherical_Harmonics::Spherical_Harm( l, m, phi, theta );
            auto R1 = Spherical_Harmonics::R( l, m, r, phi, theta );
            auto S1 = Spherical_Harmonics::S( l, m, r, phi, theta );
            auto Y2 = Spherical_Harmonics::Spherical_Harm( l, -m, phi, theta );
            auto R2 = Spherical_Harmonics::R( l, -m, r, phi, theta );
            auto S2 = Spherical_Harmonics::S( l, -m, r, phi, theta );

            REQUIRE( std::abs( Y1 - minus_one_power( m ) * std::conj( Y2 ) ) < epsilon );
            REQUIRE( std::abs( R1 - minus_one_power( m ) * std::conj( R2 ) ) < epsilon );
            REQUIRE( std::abs( S1 - minus_one_power( m ) * std::conj( S2 ) ) < epsilon );
        }
    }

    // Test against hard coded values
    // l=4, m=-1 at phi = 1.6 and theta = 0.78
    auto expected = std::complex<double>( -0.00371426, -0.127148 );
    auto result   = Spherical_Harmonics::Spherical_Harm( 4, -1, 1.6, 0.78 );
    std::cout << "expected: " << expected << std::endl;
    std::cout << "result:   " << result << std::endl;
    REQUIRE( std::abs( result - expected ) < 1e-4 );

    // l=8, m=8 at phi = 0.02 and theta = 0.5
    expected = std::complex<double>( 0.00142022, 0.000229194 );
    result   = Spherical_Harmonics::Spherical_Harm( 8, 8, 0.02, 0.5 );
    std::cout << "expected: " << expected << std::endl;
    std::cout << "result:   " << result << std::endl;
    REQUIRE( std::abs( result - expected ) < 1e-4 );

    // l=10, m=0 phi=xxx and theta=0.25
    expected = std::complex<double>( -0.141577, 0.0 );
    result   = Spherical_Harmonics::Spherical_Harm( 10, 0, 0.01212, 0.25 );
    std::cout << "expected: " << expected << std::endl;
    std::cout << "result:   " << result << std::endl;
    REQUIRE( std::abs( result - expected ) < 1e-4 );
}

TEST_CASE( "FMM", "[Translation R2R]" )
{
    scalar r     = 7.2;
    scalar theta = 1.2;
    scalar phi   = 2.2;
    int l_max    = 12;
    int l_min    = 0;

    // Test translation formulas
    Vector3 r1 = { 0.5, 0.8, 3 };
    Vector3 r2 = { 0.1, 2, -3 };

    Vector3 sph_plus;
    get_spherical( r1 + r2, sph_plus );

    Vector3 sph_minus;
    get_spherical( r1 - r2, sph_minus );

    Vector3 sph1;
    get_spherical( r1, sph1 );

    Vector3 sph2;
    get_spherical( r2, sph2 );

    // R2R
    std::cout << "=== Testing R2R ===" << std::endl;
    for( int l = 0; l <= l_max; l++ )
    {
        for( int m = -l; m <= l; m++ )
        {
            // We want to calculate R(r1 + r2) from R(r1)
            auto expected = Spherical_Harmonics::R( l, m, sph_plus[0], sph_plus[1], sph_plus[2] );
            auto result   = std::complex<double>( 0, 0 );
            for( int lp = 0; lp <= l; lp++ )
            {
                for( int mp = -lp; mp <= lp; mp++ )
                {
                    if( l - lp >= std::abs( m - mp ) )
                        result += Spherical_Harmonics::R( lp, mp, sph1[0], sph1[1], sph1[2] )
                                  * Spherical_Harmonics::R( l - lp, m - mp, sph2[0], sph2[1], sph2[2] );
                }
            }
            printf( "l %i m %i\n", l, m );
            std::cout << "expected: " << expected << std::endl;
            std::cout << "result:   " << result << std::endl;
            REQUIRE( std::abs( result - expected ) < 1e-4 );
        }
    }
}

TEST_CASE( "FMM", "[Translation S2S]" )
{
    int l_max = 12;
    int l_min = 0;

    // Test translation formulas
    Vector3 r1 = { 6, 4, 7 };
    Vector3 r2 = { 0.1, 0.2, -0.3 };

    Vector3 sph_plus;
    get_spherical( r1 + r2, sph_plus );

    Vector3 sph_minus;
    get_spherical( r1 - r2, sph_minus );

    Vector3 sph1;
    get_spherical( r1, sph1 );

    Vector3 sph2;
    get_spherical( r2, sph2 );

    // S2S
    std::cout << "=== Testing S2S ===" << std::endl;
    for( int l = 0; l <= l_max; l++ )
    {
        for( int m = -l; m <= l; m++ )
        {
            // We want to calculate S(r1 - r2) from S(r1)
            auto expected = Spherical_Harmonics::S( l, m, sph_plus[0], sph_plus[1], sph_plus[2] );
            auto result   = std::complex<double>( 0, 0 );
            for( int lp = 0; lp <= 5 * l_max; lp++ )
            {
                // for(int mp = std::max(-lp, m+lp-l); mp <= std::min(lp, m+l-lp); mp++) //because we need |m-mc| <= l-lc
                for( int mp = -lp; mp <= lp; mp++ )
                {
                    result += minus_one_power( lp + mp ) * Spherical_Harmonics::R( lp, mp, sph2[0], sph2[1], sph2[2] )
                              * Spherical_Harmonics::S( l + lp, m - mp, sph1[0], sph1[1], sph1[2] );
                }
            }
            printf( "l=%i, m = %i\n", l, m );
            std::cout << "expected: " << expected << std::endl;
            std::cout << "result:   " << result << std::endl;
            std::cout << result / expected << std::endl;
            REQUIRE( std::abs( result - expected ) < 1e-4 );
        }
    }
}