#include <catch.hpp>
#include <engine/Manifoldmath.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Vectormath_Defines.hpp>

TEST_CASE( "Manifold operations", "[manifold]" )
{
    int N       = 10000;
    int N_check = std::min( 100, N );
    vectorfield v1( N, Vector3{ 0.0, 0.0, 1.0 } );
    vectorfield v2( N, Vector3{ 1.0, 1.0, 1.0 } );

    REQUIRE( Engine::Vectormath::dot( v1, v2 ) == Approx( N ) );
    Engine::Manifoldmath::normalize( v1 );
    Engine::Manifoldmath::normalize( v2 );

    SECTION( "Normalisation" )
    {
        REQUIRE( Engine::Vectormath::dot( v1, v1 ) == Approx( 1 ) );
        REQUIRE( Engine::Manifoldmath::norm( v1 ) == Approx( 1 ) );
        REQUIRE( Engine::Manifoldmath::norm( v2 ) == Approx( 1 ) );
    }

    SECTION( "Projection: parallel" )
    {
        REQUIRE_FALSE(
            Engine::Vectormath::dot( v1, v2 )
            == Approx( Engine::Manifoldmath::norm( v1 ) * Engine::Manifoldmath::norm( v2 ) ) );
        Engine::Manifoldmath::project_parallel( v1, v2 );
        REQUIRE(
            Engine::Vectormath::dot( v1, v2 )
            == Approx( Engine::Manifoldmath::norm( v1 ) * Engine::Manifoldmath::norm( v2 ) ) );
    }

    SECTION( "Projection: orthogonal" )
    {
        REQUIRE_FALSE( Engine::Vectormath::dot( v1, v2 ) == Approx( 0 ) );
        Engine::Manifoldmath::project_orthogonal( v1, v2 );
        REQUIRE( Engine::Vectormath::dot( v1, v2 ) == Approx( 0 ) );
    }

    SECTION( "Invert: parallel" )
    {
        scalar proj_prev = Engine::Vectormath::dot( v1, v2 );
        REQUIRE_FALSE( Engine::Vectormath::dot( v1, v2 ) == Approx( -proj_prev ) );
        Engine::Manifoldmath::invert_parallel( v1, v2 );
        REQUIRE( Engine::Vectormath::dot( v1, v2 ) == Approx( -proj_prev ) );
    }

    SECTION( "Invert: orthogonal" )
    {
        vectorfield v3 = v1;
        Engine::Manifoldmath::project_orthogonal( v3, v2 );
        scalar proj_prev = Engine::Vectormath::dot( v1, v3 );
        REQUIRE( Engine::Vectormath::dot( v1, v3 ) == Approx( proj_prev ) );
        Engine::Manifoldmath::invert_orthogonal( v1, v2 );
        REQUIRE( Engine::Vectormath::dot( v1, v3 ) == Approx( -proj_prev ) );
    }

    SECTION( "Projection: tangetial" )
    {
        REQUIRE_FALSE( Engine::Vectormath::dot( v1, v2 ) == Approx( 0 ) ); // Assert they are not orthogonal
        Engine::Vectormath::normalize_vectors( v2 );                       // Normalize all vector3 of v2 for projection
        Engine::Manifoldmath::project_tangential( v1, v2 );
        for( int i = 0; i < N_check; i++ )
        {

            REQUIRE( v2[i].dot( v1[i] ) == Approx( 0 ) );
            REQUIRE( Engine::Vectormath::dot( v1, v2 ) == Approx( 0 ) );
        }
    }

    SECTION( "Distance on Greatcircle" )
    {
        // orthogonal vectors ( angle pi/2 between them )

        vectorfield vf3( N, Vector3{ 0.0, 0.0, 1.0 } );
        vectorfield vf4( N, Vector3{ 0.0, 1.0, 0.0 } );

        REQUIRE( Engine::Vectormath::dot( vf3, vf4 ) == Approx( 0 ) );
        Engine::Vectormath::normalize_vectors( vf3 ); // normalize components of vf3
        Engine::Vectormath::normalize_vectors( vf4 ); // normalize components of vf4

        scalar dist = Engine::Vectormath::angle( vf3[0], vf4[0] );
        REQUIRE( dist == Approx( std::acos( -1 ) * 0.5 ) ); // distance should be pi/2

        // antiparallel vectors ( angle pi between them )

        vectorfield vf5( N, Vector3{ 0.0, 1.0, 0.0 } );
        vectorfield vf6( N, Vector3{ 0.0, -1.0, 0.0 } );

        REQUIRE( Engine::Vectormath::dot( vf5, vf6 ) == Approx( -1 * N ) );
        Engine::Vectormath::normalize_vectors( vf5 ); // normalize components of vf5
        Engine::Vectormath::normalize_vectors( vf6 ); // normalize components of vf6

        scalar dist2 = Engine::Vectormath::angle( vf5[0], vf6[0] );
        REQUIRE( dist2 == Approx( std::acos( -1 ) ) ); // distance should be pi
    }

    SECTION( "Distance on Geodesic" )
    {
        vectorfield v3( N, Vector3{ 0.0, 0.0, 1.0 } );
        vectorfield v4( N, Vector3{ 0.0, 1.0, 0.0 } );

        REQUIRE( Engine::Vectormath::dot( v3, v4 ) == Approx( 0 ) );
        Engine::Manifoldmath::normalize( v3 );
        Engine::Manifoldmath::normalize( v4 );

        scalar dist    = Engine::Manifoldmath::dist_geodesic( v3, v4 );
        scalar dist_gc = Engine::Vectormath::angle( v3[0], v4[0] );
        REQUIRE( dist == Approx( sqrt( N * dist_gc * dist_gc ) ) );
    }
}