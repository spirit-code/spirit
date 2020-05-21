#include <catch.hpp>
#include <engine/Vectormath_Defines.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Manifoldmath.hpp>
#include <engine/Backend_par.hpp>


TEST_CASE( "Manifold operations", "[manifold]" )
{
	int N = 10000;
	int N_check = std::min(100, N);
	vectorfield v1(N, Vector3{ 0.0, 0.0, 1.0 });
	vectorfield v2(N, Vector3{ 1.0, 1.0, 1.0 });

	REQUIRE( Engine::Vectormath::dot(v1,v2) == Approx(N) );
	Engine::Manifoldmath::normalize(v1);
	Engine::Manifoldmath::normalize(v2);

	SECTION("Normalisation")
	{
		REQUIRE( Engine::Vectormath::dot(v1, v1) == Approx(1) );
		REQUIRE( Engine::Manifoldmath::norm(v1) == Approx(1) );
		REQUIRE( Engine::Manifoldmath::norm(v2) == Approx(1) );
	}

	SECTION("Projection: parallel")
	{
		REQUIRE_FALSE( Engine::Vectormath::dot(v1,v2) == Approx(Engine::Manifoldmath::norm(v1)*Engine::Manifoldmath::norm(v2)) );
		Engine::Manifoldmath::project_parallel(v1,v2);
		REQUIRE( Engine::Vectormath::dot(v1,v2) == Approx(Engine::Manifoldmath::norm(v1)*Engine::Manifoldmath::norm(v2)) );
	}

	SECTION("Projection: orthogonal")
	{
		REQUIRE_FALSE( Engine::Vectormath::dot(v1,v2) == Approx(0) );
		Engine::Manifoldmath::project_orthogonal(v1,v2);
		REQUIRE( Engine::Vectormath::dot(v1,v2) == Approx(0) );
	}

	SECTION("Invert: parallel")
	{
		scalar proj_prev = Engine::Vectormath::dot(v1, v2);
		REQUIRE_FALSE( Engine::Vectormath::dot(v1,v2) == Approx(-proj_prev) );
		Engine::Manifoldmath::invert_parallel(v1,v2);
		REQUIRE( Engine::Vectormath::dot(v1,v2) == Approx(-proj_prev) );
	}

	SECTION("Invert: orthogonal")
	{
		vectorfield v3 = v1;
		Engine::Manifoldmath::project_orthogonal(v3, v2);
		scalar proj_prev = Engine::Vectormath::dot(v1, v3);
		REQUIRE( Engine::Vectormath::dot(v1, v3) == Approx(proj_prev) );
		Engine::Manifoldmath::invert_orthogonal(v1,v2);
		REQUIRE( Engine::Vectormath::dot(v1, v3) == Approx(-proj_prev) );
	}

  SECTION( "Projection: tangetial")
  {
    REQUIRE_FALSE( Engine::Vectormath::dot(v1,v2) == Approx(0) );  // Assert they are not orthogonal
    // Normalize all vector3 of v2 for projection
    auto v2p = v2.data();
    Engine::Backend::par::apply(v2.size(), [v2p] SPIRIT_LAMBDA (int idx) {
        v2p[idx].normalize();
    });
    Engine::Manifoldmath::project_tangential(v1,v2);
    for (int i=0; i<N_check; i++)
    {

      REQUIRE( v2[i].dot(v1[i]) == Approx(0) );
      REQUIRE( Engine::Vectormath::dot(v1,v2) == Approx(0) );
    }
  }

  SECTION( "Distance on Greatcircle" )
  {
    // orthogonal vectors ( angle pi/2 between them )

    vectorfield vf3( N, Vector3{ 0.0, 0.0, 1.0 } );
    vectorfield vf4( N, Vector3{ 0.0, 1.0, 0.0 } );

    REQUIRE( Engine::Vectormath::dot( vf3, vf4) == Approx(0) );
    auto vf3p = vf3.data();
    auto vf4p = vf4.data();
    Engine::Backend::par::apply(vf3.size(), [vf3p, vf4p] SPIRIT_LAMBDA (int idx) {
        vf3p[idx].normalize();
        vf4p[idx].normalize();
    });

    scalar dist = Engine::Vectormath::angle( vf3[0], vf4[0] );
    REQUIRE( dist == Approx( std::acos(-1)*0.5 ) );   // distance should be pi/2

    // antiparallel vectors ( angle pi between them )

    vectorfield vf5( N, Vector3{ 0.0,  1.0, 0.0 } );
    vectorfield vf6( N, Vector3{ 0.0, -1.0, 0.0 } );

    REQUIRE( Engine::Vectormath::dot( vf5, vf6 ) == Approx(-1*N) );
    auto vf5p = vf5.data();
    auto vf6p = vf6.data();
    Engine::Backend::par::apply(vf5.size(), [vf5p, vf6p] SPIRIT_LAMBDA (int idx) {
        vf5p[idx].normalize();
        vf6p[idx].normalize();
    });

    scalar dist2 = Engine::Vectormath::angle( vf5[0], vf6[0] );
    REQUIRE( dist2 == Approx( std::acos(-1) ) );      // distance should be pi
  }

  SECTION( "Distance on Geodesic" )
  {
    vectorfield v3( N, Vector3{ 0.0, 0.0, 1.0 } );
    vectorfield v4( N, Vector3{ 0.0, 1.0, 0.0 } );

    REQUIRE( Engine::Vectormath::dot( v3, v4 ) == Approx( 0 ) );
    Engine::Manifoldmath::normalize( v3 );
    Engine::Manifoldmath::normalize( v4 );

    scalar dist = Engine::Manifoldmath::dist_geodesic( v3, v4 );
    scalar dist_gc = Engine::Vectormath::angle( v3[0], v4[0] );
    REQUIRE( dist == Approx( sqrt( N * dist_gc * dist_gc ) ) );
  }
}