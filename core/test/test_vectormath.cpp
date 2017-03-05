#include <catch.hpp>
#include <engine/Vectormath_Defines.hpp>
#include <engine/Vectormath.hpp>


TEST_CASE( "Vectormath operations", "[vectormath]" )
{
    int N = 10000;
    vectorfield v1(N, Vector3{ 1.0, 1.0, 1.0 });
    vectorfield v2(N, Vector3{ -1.0, 1.0, 1.0 });
	
	SECTION("Dot Product")
	{
		scalarfield dots(N, N);
		Engine::Vectormath::dot(v1,v2, dots);
		REQUIRE( dots[0] == Approx(1) );
		REQUIRE(Engine::Vectormath::dot(v1,v2) == Approx(N) );
	}

	// SECTION("Add c*a")
	// {
	// }
	// SECTION("Add c*v1.dot(v2)")
	// {
	// }
	// SECTION("Add c*v1.cross(v2)")
	// {
	// }
}