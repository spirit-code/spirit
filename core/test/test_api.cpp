#include <catch.hpp>
#include <interface/Interface_State.h>
#include <interface/Interface_Configurations.h>
#include <interface/Interface_Quantities.h>


TEST_CASE( "State", "[state]" )
{
	CHECK_NOTHROW(std::shared_ptr<State>(State_Setup(), State_Delete));
	CHECK_NOTHROW(std::shared_ptr<State>(State_Setup("input/input.txt"), State_Delete));
}

TEST_CASE( "Configurations", "[configurations]" )
{

}

TEST_CASE( "Quantities", "[quantities]" )
{
	SECTION("Magnetization")
	{
		auto state = std::shared_ptr<State>(State_Setup("input/input.txt"), State_Delete);

		float m[3];
		float min[3] = { -1e8, -1e8, -1e8 };

		SECTION("001")
		{
			float dir[3] = { 0,0,1 };

			Configuration_DomainWall(state.get(), min, dir, true);
			Quantity_Get_Magnetization(state.get(), m);

			REQUIRE(m[0] == Approx(dir[0]));
			REQUIRE(m[1] == Approx(dir[1]));
			REQUIRE(m[2] == Approx(dir[2]));
		}
		SECTION("011")
		{
			float dir[3] = { 0,0,1 };

			Configuration_DomainWall(state.get(), min, dir, true);
			Quantity_Get_Magnetization(state.get(), m);

			REQUIRE(m[0] == Approx(dir[0]));
			REQUIRE(m[1] == Approx(dir[1]));
			REQUIRE(m[2] == Approx(dir[2]));
		}
	}

}