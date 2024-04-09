#include <engine/Manifoldmath.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Vectormath_Defines.hpp>

#include "catch.hpp"

using Catch::Matchers::WithinAbs;

auto inputfile = "core/test/input/api.cfg";
// Reduce required precision if float accuracy
#ifdef SPIRIT_SCALAR_TYPE_DOUBLE
[[maybe_unused]] constexpr scalar epsilon_2 = 1e-10;
[[maybe_unused]] constexpr scalar epsilon_3 = 1e-12;
[[maybe_unused]] constexpr scalar epsilon_4 = 1e-12;
[[maybe_unused]] constexpr scalar epsilon_5 = 1e-12;
[[maybe_unused]] constexpr scalar epsilon_6 = 1e-12;
[[maybe_unused]] constexpr scalar epsilon_7 = 1e-12;
#else
[[maybe_unused]] constexpr scalar epsilon_2 = 1e-2;
[[maybe_unused]] constexpr scalar epsilon_3 = 1e-3;
[[maybe_unused]] constexpr scalar epsilon_4 = 1e-4;
[[maybe_unused]] constexpr scalar epsilon_5 = 1e-5;
[[maybe_unused]] constexpr scalar epsilon_6 = 1e-6;
[[maybe_unused]] constexpr scalar epsilon_7 = 1e-7;
#endif

TEST_CASE( "Manifold operations", "[manifold]" )
{
    int N       = 10000;
    int N_check = std::min( 100, N );
    vectorfield v1( N, Vector3{ 0.0, 0.0, 1.0 } );
    vectorfield v2( N, Vector3{ 1.0, 1.0, 1.0 } );

    REQUIRE_THAT( Engine::Vectormath::dot( v1, v2 ), WithinAbs( N, 1e-12 ) );
    Engine::Manifoldmath::normalize( v1 );
    Engine::Manifoldmath::normalize( v2 );

    SECTION( "Normalisation" )
    {
        REQUIRE_THAT( Engine::Vectormath::dot( v1, v1 ), WithinAbs( 1, epsilon_4 ) );
        REQUIRE_THAT( Engine::Manifoldmath::norm( v1 ), WithinAbs( 1, epsilon_4 ) );
        REQUIRE_THAT( Engine::Manifoldmath::norm( v2 ), WithinAbs( 1, epsilon_4 ) );
    }

    SECTION( "Projection: parallel" )
    {
        REQUIRE_THAT(
            Engine::Vectormath::dot( v1, v2 ),
            !WithinAbs( Engine::Manifoldmath::norm( v1 ) * Engine::Manifoldmath::norm( v2 ), 1e-3 ) );
        Engine::Manifoldmath::project_parallel( v1, v2 );
        REQUIRE_THAT(
            Engine::Vectormath::dot( v1, v2 ),
            WithinAbs( Engine::Manifoldmath::norm( v1 ) * Engine::Manifoldmath::norm( v2 ), epsilon_3 ) );
    }

    SECTION( "Projection: orthogonal" )
    {
        REQUIRE_THAT( Engine::Vectormath::dot( v1, v2 ), !WithinAbs( 0, 1e-3 ) );
        Engine::Manifoldmath::project_orthogonal( v1, v2 );
        REQUIRE_THAT( Engine::Vectormath::dot( v1, v2 ), WithinAbs( 0, epsilon_5 ) );
    }

    SECTION( "Invert: parallel" )
    {
        scalar proj_prev = Engine::Vectormath::dot( v1, v2 );
        REQUIRE_THAT( Engine::Vectormath::dot( v1, v2 ), !WithinAbs( -proj_prev, 1e-3 ) );
        Engine::Manifoldmath::invert_parallel( v1, v2 );
        REQUIRE_THAT( Engine::Vectormath::dot( v1, v2 ), WithinAbs( -proj_prev, epsilon_4 ) );
    }

    SECTION( "Invert: orthogonal" )
    {
        vectorfield v3 = v1;
        Engine::Manifoldmath::project_orthogonal( v3, v2 );
        scalar proj_prev = Engine::Vectormath::dot( v1, v3 );
        REQUIRE_THAT( Engine::Vectormath::dot( v1, v3 ), WithinAbs( proj_prev, 1e-12 ) );
        Engine::Manifoldmath::invert_orthogonal( v1, v2 );
        REQUIRE_THAT( Engine::Vectormath::dot( v1, v3 ), WithinAbs( -proj_prev, epsilon_6 ) );
    }

    SECTION( "Projection: tangetial" )
    {
        REQUIRE_THAT( Engine::Vectormath::dot( v1, v2 ), !WithinAbs( 0, 1e-3 ) ); // Assert they are not orthogonal
        Engine::Vectormath::normalize_vectors( v2 ); // Normalize all vector3 of v2 for projection
        Engine::Manifoldmath::project_tangential( v1, v2 );
        for( int i = 0; i < N_check; i++ )
        {
            REQUIRE_THAT( v2[i].dot( v1[i] ), WithinAbs( 0, epsilon_7 ) );
            REQUIRE_THAT( Engine::Vectormath::dot( v1, v2 ), WithinAbs( 0, epsilon_5 ) );
        }
    }

    SECTION( "Distance on Greatcircle" )
    {
        // Orthogonal vectors ( angle pi/2 between them )

        vectorfield vf3( N, Vector3{ 0.0, 0.0, 1.0 } );
        vectorfield vf4( N, Vector3{ 0.0, 1.0, 0.0 } );

        REQUIRE_THAT( Engine::Vectormath::dot( vf3, vf4 ), WithinAbs( 0, 1e-12 ) );
        Engine::Vectormath::normalize_vectors( vf3 ); // Normalize components of vf3
        Engine::Vectormath::normalize_vectors( vf4 ); // Normalize components of vf4

        scalar dist = Engine::Vectormath::angle( vf3[0], vf4[0] );
        REQUIRE_THAT( dist, WithinAbs( std::acos( -1 ) * 0.5, epsilon_7 ) ); // Distance should be pi/2

        // Antiparallel vectors ( angle pi between them )

        vectorfield vf5( N, Vector3{ 0.0, 1.0, 0.0 } );
        vectorfield vf6( N, Vector3{ 0.0, -1.0, 0.0 } );

        REQUIRE_THAT( Engine::Vectormath::dot( vf5, vf6 ), WithinAbs( -1 * N, 1e-12 ) );
        Engine::Vectormath::normalize_vectors( vf5 ); // Normalize components of vf5
        Engine::Vectormath::normalize_vectors( vf6 ); // Normalize components of vf6

        scalar dist2 = Engine::Vectormath::angle( vf5[0], vf6[0] );
        REQUIRE_THAT( dist2, WithinAbs( std::acos( -1 ), epsilon_6 ) ); // Distance should be pi
    }

    SECTION( "Distance on Geodesic" )
    {
        vectorfield v3( N, Vector3{ 0.0, 0.0, 1.0 } );
        vectorfield v4( N, Vector3{ 0.0, 1.0, 0.0 } );

        REQUIRE_THAT( Engine::Vectormath::dot( v3, v4 ), WithinAbs( 0, 1e-12 ) );
        Engine::Manifoldmath::normalize( v3 );
        Engine::Manifoldmath::normalize( v4 );

        scalar dist    = Engine::Manifoldmath::dist_geodesic( v3, v4 );
        scalar dist_gc = Engine::Vectormath::angle( v3[0], v4[0] );
        REQUIRE_THAT( dist, WithinAbs( sqrt( N * dist_gc * dist_gc ), epsilon_2 ) );
    }
}
