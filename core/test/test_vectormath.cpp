#include <catch.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Vectormath_Defines.hpp>

TEST_CASE( "Vectormath operations", "[vectormath]" )
{
    int N       = 10000;
    int N_check = std::min( 100, N );
    scalarfield sf( N, 1 );
    vectorfield vf1( N, Vector3{ 1.0, 1.0, 1.0 } );
    vectorfield vf2( N, Vector3{ -1.0, 1.0, 1.0 } );

    SECTION( "Magnetization" )
    {
        Engine::Vectormath::fill( vf1, { 0, 0, 1 } );
        auto m = Engine::Vectormath::Magnetization( vf1 );
        REQUIRE( m[0] == Approx( 0 ) );
        REQUIRE( m[1] == Approx( 0 ) );
        REQUIRE( m[2] == Approx( 1 ) );
    }

    SECTION( "Rotate" )
    {
        scalar pi = std::acos( -1 );

        Vector3 v1_out{ 0, 0, 0 };
        Vector3 v1_in{ 1, 0, 1 };
        Vector3 v1_axis{ 0, 0, 1 };
        Vector3 v1_exp{ 0, 1, 1 }; // expected result
        scalar angle = pi / 2;

        Engine::Vectormath::rotate( v1_in, v1_axis, angle, v1_out );
        for( unsigned int i = 0; i < 3; i++ )
            REQUIRE( v1_out[i] == Approx( v1_exp[i] ) );

        // zero rotation test
        Vector3 v2_out{ 0, 0, 0 };
        Vector3 v2_in{ 1, 1, 1 };
        Vector3 v2_axis{ 1, 0, 0 };
        Vector3 v2_exp{ 1, 1, 1 };
        scalar angle2 = 0;

        Engine::Vectormath::rotate( v2_in, v2_axis, angle2, v2_out );
        for( unsigned int i = 0; i < 3; i++ )
            REQUIRE( v2_out[i] == Approx( v2_exp[i] ) );
    }

    SECTION( "Fill" )
    {
        scalar stest = 333;
        Vector3 vtest{ 0, stest, 0 };
        Engine::Vectormath::fill( sf, stest );
        Engine::Vectormath::fill( vf1, vtest );
        for( int i = 0; i < N_check; ++i )
        {
            REQUIRE( sf[i] == stest );
            REQUIRE( vf1[i] == vtest );
        }
    }

    SECTION( "Scale" )
    {
        scalar stest = 555;
        Vector3 vtest{ stest, stest, stest };
        Engine::Vectormath::scale( sf, stest );
        Engine::Vectormath::scale( vf1, stest );
        for( int i = 0; i < N_check; ++i )
        {
            REQUIRE( sf[i] == stest );
            REQUIRE( vf1[i] == vtest );
        }
    }

    SECTION( "Sum, Mean and Divide" )
    {
        // Sum
        scalar sN = (scalar)N;
        Vector3 vref1{ sN, sN, sN };
        Vector3 vref2{ -sN, sN, sN };
        scalar stest1  = Engine::Vectormath::sum( sf );
        Vector3 vtest1 = Engine::Vectormath::sum( vf1 );
        Vector3 vtest2 = Engine::Vectormath::sum( vf2 );
        REQUIRE( stest1 == sN );
        REQUIRE( vtest1 == vref1 );
        REQUIRE( vtest2 == vref2 );

        // Mean
        Vector3 vref3{ 1.0, 1.0, 1.0 };
        Vector3 vref4{ -1.0, 1.0, 1.0 };
        scalar stest2  = Engine::Vectormath::mean( sf );
        Vector3 vtest3 = Engine::Vectormath::mean( vf1 );
        Vector3 vtest4 = Engine::Vectormath::mean( vf2 );
        REQUIRE( stest2 == 1 );
        REQUIRE( vtest3 == vref3 );
        REQUIRE( vtest4 == vref4 );

        // Divide
        scalar stest3 = 3;
        scalarfield numerator( N, 6 );
        scalarfield denominator( N, 2 );
        scalarfield result( N, 0 );
        Engine::Vectormath::divide( numerator, denominator, result );
        for( int i = 0; i < N_check; i++ )
        {
            REQUIRE( result[i] == stest3 );
        }
    }

    SECTION( "Normalization" )
    {
        scalar mc = Engine::Vectormath::max_abs_component( vf1 );
        REQUIRE( mc == 1 );

        Vector3 vtest1 = vf1[0].normalized();
        Vector3 vtest2 = vf2[0].normalized();
        Engine::Vectormath::normalize_vectors( vf1 );
        Engine::Vectormath::normalize_vectors( vf2 );
        for( int i = 0; i < N_check; ++i )
        {
            REQUIRE( vf1[i] == vtest1 );
            REQUIRE( vf2[i] == vtest2 );
        }
    }

    SECTION( "MAX Abs component" )
    {
        Vector3 vtest1 = vf1[0].normalized();
        Vector3 vtest2 = vf2[0].normalized();
        scalar vmc1    = std::max( vtest1[0], std::max( vtest1[1], vtest1[2] ) );
        scalar vmc2    = std::max( vtest2[0], std::max( vtest2[1], vtest2[2] ) );

        Engine::Vectormath::normalize_vectors( vf1 );
        Engine::Vectormath::normalize_vectors( vf2 );
        scalar vfmc1 = Engine::Vectormath::max_abs_component( vf1 );
        scalar vfmc2 = Engine::Vectormath::max_abs_component( vf2 );

        REQUIRE( vfmc1 == vmc1 );
        REQUIRE( vfmc2 == vmc2 );
    }

    SECTION( "MAX norm" )
    {
        scalar norm_test = vf1[0].norm();
        scalar max_norm  = Engine::Vectormath::max_norm( vf1 );

        REQUIRE( norm_test == max_norm );
    }

    SECTION( "Dot and Cross Product" )
    {
        // Dot Product
        scalarfield dots( N, N );
        Engine::Vectormath::dot( vf1, vf2, dots );
        REQUIRE( dots[0] == Approx( 1 ) );
        REQUIRE( Engine::Vectormath::dot( vf1, vf2 ) == Approx( N ) );

        // Scalarfields dot Product
        scalarfield sf1( N, 2 );
        scalarfield sf2( N, -0.5 );
        scalarfield result( N, 0 );
        Engine::Vectormath::dot( sf1, sf2, result );
        for( int i = 0; i < N; i++ )
            REQUIRE( result[i] == -1 );

        // Cross Product
        Vector3 vtest{ 0, -2, 2 };
        vectorfield crosses( N );
        Engine::Vectormath::cross( vf1, vf2, crosses );
        REQUIRE( crosses[0] == vtest );
    }

    SECTION( "c*a" )
    {
        // out[i] += c*a
        Vector3 vtest1{ 1.0, 1.0, 1.0 };
        Vector3 vtest2{ 1.0, 3.0, 3.0 };
        Engine::Vectormath::add_c_a( 2, vtest1, vf2 );
        for( int i = 0; i < N_check; ++i )
            REQUIRE( vf2[i] == vtest2 );

        // out[i] += c*a[i]
        Vector3 vtest3{ 0.0, -2.0, -2.0 };
        Engine::Vectormath::add_c_a( -1, vf2, vf1 );
        for( int i = 0; i < N_check; ++i )
            REQUIRE( vf1[i] == vtest3 );

        // out[i] = c*a
        Engine::Vectormath::set_c_a( 3, vtest1, vf1 ); // vf1 is now { 3, 3, 3 }
        for( int i = 0; i < N_check; ++i )
            REQUIRE( vf1[i] == 3 * vtest1 );

        // out[i] = c*a[i]
        Engine::Vectormath::set_c_a( 3, vf1, vf2 ); // vf2 is now { 9, 9, 9 }
        for( int i = 0; i < N_check; ++i )
            REQUIRE( vf2[i] == 3 * vf1[i] );

        // out[i] += c[i]*a[i]
        Vector3 vtest4{ -6, -6, -6 };
        scalarfield sf( N, -1 );
        Engine::Vectormath::add_c_a( sf, vf2, vf1 ); // vf1 is now { -6, -6, -6 }
        for( int i = 0; i < N_check; i++ )
            REQUIRE( vf1[i] == vtest4 );

        // out[i] = c[i]*a[i]
        Vector3 vtest5{ 6, 6, 6 };
        Engine::Vectormath::set_c_a( sf, vf1, vf2 ); // vf2 is now { 6, 6, 6 }
        for( int i = 0; i < N_check; i++ )
            REQUIRE( vf2[i] == vtest5 );
    }

    SECTION( "c*v1.dot(v2)" )
    {
        // out[i] += c * a*b[i]
        Vector3 vtest1{ 1.0, -2.0, -3.0 };
        Engine::Vectormath::add_c_dot( -3, vtest1, vf1, sf );
        for( int i = 0; i < N_check; ++i )
            REQUIRE( sf[i] == 13 );

        // out[i] += c * a[i]*b[i]
        Engine::Vectormath::add_c_dot( -2, vf1, vf2, sf );
        for( int i = 0; i < N_check; ++i )
            REQUIRE( sf[i] == 11 );

        // out[i] = c * a*b[i]
        Engine::Vectormath::set_c_dot( 3, vtest1, vf1, sf );
        for( int i = 0; i < N_check; ++i )
            REQUIRE( sf[i] == -12 );

        // out[i] = c * a[i]*b[i]
        Engine::Vectormath::set_c_dot( 2, vf1, vf2, sf );
        for( int i = 0; i < N_check; ++i )
            REQUIRE( sf[i] == 2 );
    }

    SECTION( "c*v1.cross(v2)" )
    {
        vectorfield vftest( N, Vector3{ 1.0, 1.0, 1.0 } );

        // out[i] += c * a x b[i]
        Vector3 vtest1{ 1.0, 9.0, -7.0 };
        Engine::Vectormath::add_c_cross( 4, vf2[0], vf1, vftest );
        for( int i = 0; i < N_check; ++i )
            REQUIRE( vftest[i] == vtest1 );

        // out[i] += c * a[i] x b[i]
        Vector3 vtest2{ 1.0, 1.0, 1.0 };
        Engine::Vectormath::add_c_cross( 4, vf1, vf2, vftest );
        for( int i = 0; i < N_check; ++i )
            REQUIRE( vftest[i] == vtest2 );

        // out[i] = c * a x b[i]
        Vector3 vtest3{ 0.0, -6.0, 6.0 };
        Engine::Vectormath::set_c_cross( 3, vf1[0], vf2, vftest );
        for( int i = 0; i < N_check; ++i )
            REQUIRE( vftest[i] == vtest3 );

        // out[i] = c * a[i] x b[i]
        Engine::Vectormath::set_c_cross( 3, vf1, vf2, vftest );
        for( int i = 0; i < N_check; ++i )
            REQUIRE( vftest[i] == vtest3 );
    }
}