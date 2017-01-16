#include <catch.hpp>
#include <engine/Vectormath_Defines.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Manifoldmath.hpp>


TEST_CASE( "Manifold operations", "[manifold]" )
{
	int N = 10000;
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
}