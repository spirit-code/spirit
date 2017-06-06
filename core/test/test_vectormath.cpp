#include <catch.hpp>
#include <engine/Vectormath_Defines.hpp>
#include <engine/Vectormath.hpp>


TEST_CASE( "Vectormath operations", "[vectormath]" )
{
    int N = 10000;
	scalarfield sf(N, 1);
    vectorfield vf1(N, Vector3{ 1.0, 1.0, 1.0 });
    vectorfield vf2(N, Vector3{ -1.0, 1.0, 1.0 });
	
	SECTION("Fill")
	{
		scalar stest = 333;
		Vector3 vtest{ 0, stest, 0 };
		Engine::Vectormath::fill(sf, stest);
		Engine::Vectormath::fill(vf1, vtest);
		for (int i = 0; i < N; ++i)
		{
			REQUIRE(sf[i] == stest);
			REQUIRE(vf1[i] == vtest);
		}
	}

	SECTION("Scale")
	{
		scalar stest = 555;
		Vector3 vtest{ stest, stest, stest };
		Engine::Vectormath::scale(sf, stest);
		Engine::Vectormath::scale(vf1, stest);
		for (int i = 0; i < N; ++i)
		{
			REQUIRE(sf[i] == stest);
			REQUIRE(vf1[i] == vtest);
		}
	}

	SECTION("Sum and Mean")
	{
		// Sum
		scalar sN = (scalar)N;
		Vector3 vref1{ sN, sN, sN };
		Vector3 vref2{ -sN, sN, sN };
		scalar  stest1 = Engine::Vectormath::sum(sf);
		Vector3 vtest1 = Engine::Vectormath::sum(vf1);
		Vector3 vtest2 = Engine::Vectormath::sum(vf2);
		REQUIRE(stest1 == sN);
		REQUIRE(vtest1 == vref1);
		REQUIRE(vtest2 == vref2);

		// Mean
		Vector3 vref3{  1.0, 1.0, 1.0 };
		Vector3 vref4{ -1.0, 1.0, 1.0 };
		scalar  stest2 = Engine::Vectormath::mean(sf);
		Vector3 vtest3 = Engine::Vectormath::mean(vf1);
		Vector3 vtest4 = Engine::Vectormath::mean(vf2);
		REQUIRE(stest2 == 1);
		REQUIRE(vtest3 == vref3);
		REQUIRE(vtest4 == vref4);
	}

	SECTION("Normalization")
	{
		scalar mc = Engine::Vectormath::max_abs_component(vf1);
		REQUIRE(mc == 1);
    
    Vector3 vtest1 = vf1[0].normalized();
    Vector3 vtest2 = vf2[0].normalized();
    Engine::Vectormath::normalize_vectors(vf1);
    Engine::Vectormath::normalize_vectors(vf2);
    for (int i = 0; i < N; ++i)
    {
      REQUIRE(vf1[i] == vtest1);
      REQUIRE(vf2[i] == vtest2);
    }
  }
  
  SECTION( "MIN and MAX components" )
  {
    vectorfield vftest( N, Vector3{ 1., 1., 1. } );
    vftest[0][1] = 10000;
    vftest[N/2][2] = -10000;
    std::pair< scalar, scalar > mm = Engine::Vectormath::minmax_component( vftest );
    REQUIRE( mm.first == -10000 );    // check min
    REQUIRE( mm.second == 10000 );    // check max
  }
  
  SECTION( "MAX Abs component" )
  {
    Vector3 vtest1 = vf1[0].normalized();
    Vector3 vtest2 = vf2[0].normalized();
    scalar vmc1 = std::max( vtest1[0], std::max( vtest1[1], vtest1[2] ) );
    scalar vmc2 = std::max( vtest2[0], std::max( vtest2[1], vtest2[2] ) );
    
    Engine::Vectormath::normalize_vectors(vf1);
    Engine::Vectormath::normalize_vectors(vf2);
    scalar vfmc1 = Engine::Vectormath::max_abs_component( vf1 );
    scalar vfmc2 = Engine::Vectormath::max_abs_component( vf2 );
    
    REQUIRE( vfmc1 == vmc1 );
    REQUIRE( vfmc2 == vmc2 );
  }
  
	SECTION("Dot and Cross Product")
	{
		// Dot Product
		scalarfield dots(N, N);
		Engine::Vectormath::dot(vf1, vf2, dots);
		REQUIRE( dots[0] == Approx(1) );
		REQUIRE( Engine::Vectormath::dot(vf1, vf2) == Approx(N) );

		// Cross Product
		Vector3 vtest{ 0, -2, 2 };
		vectorfield crosses(N);
		Engine::Vectormath::cross(vf1, vf2, crosses);
		REQUIRE( crosses[0] == vtest );
	}

	SECTION("c*a")
	{
		// out[i] += c*a
		Vector3 vtest1{ 1.0, 1.0, 1.0 };
		Vector3 vtest2{ 1.0, 3.0, 3.0 };
		Engine::Vectormath::add_c_a(2, vtest1, vf2);
		for (int i = 0; i < N; ++i)
		{
			REQUIRE(vf2[i] == vtest2);
		}
		// out[i] += c*a[i]
		Vector3 vtest3{ 0.0, -2.0, -2.0 };
		Engine::Vectormath::add_c_a(-1, vf2, vf1);
		for (int i = 0; i < N; ++i)
		{
			REQUIRE(vf1[i] == vtest3);
		}
		// out[i] = c*a
		Engine::Vectormath::set_c_a(3, vtest1, vf1);
		for (int i = 0; i < N; ++i)
		{
			REQUIRE(vf1[i] == 3*vtest1);
		}
		// out[i] = c*a[i]
		Engine::Vectormath::set_c_a(3, vf1, vf2);
		for (int i = 0; i < N; ++i)
		{
			REQUIRE(vf2[i] == 3*vf1[i]);
		}
	}
	
	SECTION("c*v1.dot(v2)")
	{
		// out[i] += c * a*b[i]
		Vector3 vtest1{ 1.0, -2.0, -3.0 };
		Engine::Vectormath::add_c_dot(-3, vtest1, vf1, sf);
		for (int i = 0; i < N; ++i)
		{
			REQUIRE(sf[i] == 13);
		}
		// out[i] += c * a[i]*b[i]
		Engine::Vectormath::add_c_dot(-2, vf1, vf2, sf);
		for (int i = 0; i < N; ++i)
		{
			REQUIRE(sf[i] == 11);
		}

		// out[i] = c * a*b[i]
		Engine::Vectormath::set_c_dot(3, vtest1, vf1, sf);
		for (int i = 0; i < N; ++i)
		{
			REQUIRE(sf[i] == -12);
		}
		// out[i] = c * a[i]*b[i]
		Engine::Vectormath::set_c_dot(2, vf1, vf2, sf);
		for (int i = 0; i < N; ++i)
		{
			REQUIRE(sf[i] == 2);
		}
	}
	
	SECTION("c*v1.cross(v2)")
	{
		vectorfield vftest(N, Vector3{ 1.0, 1.0, 1.0 });

		// out[i] += c * a x b[i]
		Vector3 vtest1{ 1.0, 9.0, -7.0 };
		Engine::Vectormath::add_c_cross(4, vf2[0], vf1, vftest);
		for (int i = 0; i < N; ++i)
		{
			REQUIRE(vftest[i] == vtest1);
		}
		// out[i] += c * a[i] x b[i]
		Vector3 vtest2{ 1.0, 1.0, 1.0 };
		Engine::Vectormath::add_c_cross(4, vf1, vf2, vftest);
		for (int i = 0; i < N; ++i)
		{
			REQUIRE(vftest[i] == vtest2);
		}

		// out[i] = c * a x b[i]
		Vector3 vtest3{ 0.0, -6.0, 6.0 };
		Engine::Vectormath::set_c_cross(3, vf1[0], vf2, vftest);
		for (int i = 0; i < N; ++i)
		{
			REQUIRE(vftest[i] == vtest3);
		}
		// out[i] = c * a[i] x b[i]
		Engine::Vectormath::set_c_cross(3, vf1, vf2, vftest);
		for (int i = 0; i < N; ++i)
		{
			REQUIRE(vftest[i] == vtest3);
		}
	}
}