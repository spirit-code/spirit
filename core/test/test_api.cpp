#include <catch.hpp>
#include <Spirit/State.h>
#include <Spirit/Configurations.h>
#include <Spirit/Quantities.h>


TEST_CASE( "State", "[state]" )
{
	// Test the default config
	CHECK_NOTHROW(std::shared_ptr<State>(State_Setup(), State_Delete));
	// Test the default input file
	CHECK_NOTHROW(std::shared_ptr<State>(State_Setup("input/input.cfg"), State_Delete));
}

TEST_CASE( "Configurations", "[configurations]" )
{
	auto state = std::shared_ptr<State>(State_Setup("input/input.cfg"), State_Delete);
	
	// filters
	float position[3]{0,0,0};
	float r_cut_rectangular[3]{-1,-1,-1};
	float r_cut_cylindrical = -1;
	float r_cut_spherical = -1;
	bool inverted = false;

	SECTION("Domain")
	{
		float dir[3] = { 0,0,1 };
		Configuration_PlusZ(state.get(), position, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted);
		Configuration_MinusZ(state.get(), position, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted);
		Configuration_Domain(state.get(), dir, position, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted);
	}
	SECTION("Random")
	{
		float temperature = 5;
		Configuration_Add_Noise_Temperature(state.get(), temperature, position, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted);
		Configuration_Random(state.get(), position, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted);
	}
	SECTION("Skyrmion")
	{
		float r=5;
		int order=1;
		float phase=0;
		bool updown=false, achiral=false, rl=false;
		Configuration_Skyrmion(state.get(), r, order, phase, updown, achiral, rl, position, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted);

		r=7;
		order=1;
		phase=-90,
		updown=false; achiral=true; rl=false;
		Configuration_Skyrmion(state.get(), r, order, phase, updown, achiral, rl, position, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted);
	}
	SECTION("Hopfion")
	{
		float r=5;
		int order=1;
		Configuration_Hopfion(state.get(), r, order, position, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted);
	}
	SECTION("Spin Spiral")
	{
		auto dir_type = "real lattice";
		float q[3]{0,0,0.1}, axis[3]{0,0,1}, theta{30};
		CHECK_NOTHROW( Configuration_SpinSpiral(state.get(), dir_type, q, axis, theta, position, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted); );
	}
}

TEST_CASE( "Quantities", "[quantities]" )
{
	SECTION("Magnetization")
	{
		auto state = std::shared_ptr<State>(State_Setup("input/input.cfg"), State_Delete);
		float m[3] = { 0,0,1 };

		SECTION("001")
		{
			float dir[3] = { 0,0,1 };

			Configuration_Domain(state.get(), dir);
			Quantity_Get_Magnetization(state.get(), m);

			REQUIRE(m[0] == Approx(dir[0]));
			REQUIRE(m[1] == Approx(dir[1]));
			REQUIRE(m[2] == Approx(dir[2]));
		}
		SECTION("011")
		{
			float dir[3] = { 0,0,1 };

			Configuration_Domain(state.get(), dir);
			Quantity_Get_Magnetization(state.get(), m);

			REQUIRE(m[0] == Approx(dir[0]));
			REQUIRE(m[1] == Approx(dir[1]));
			REQUIRE(m[2] == Approx(dir[2]));
		}
	}
	SECTION("Topological Charge")
	{
		auto state = std::shared_ptr<State>(State_Setup("input/input.cfg"), State_Delete);
		
		SECTION("negative charge")
		{
			Configuration_PlusZ(state.get());
			Configuration_Skyrmion(state.get(), 6.0, 1.0, -90.0, false, false, false);
			float charge = Quantity_Get_Topological_Charge(state.get());
			REQUIRE(charge == Approx(-1));
		}

		SECTION("positive charge")
		{
			Configuration_MinusZ(state.get());
			Configuration_Skyrmion(state.get(), 6.0, 1.0, -90.0, true, false, false);
			float charge = Quantity_Get_Topological_Charge(state.get());
			REQUIRE(charge == Approx(1));
		}
	}
}