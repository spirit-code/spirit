#include <Spirit/Configurations.h>
#include <Spirit/Constants.h>
#include <Spirit/Geometry.h>
#include <Spirit/Hamiltonian.h>
#include <Spirit/Parameters_LLG.h>
#include <Spirit/Simulation.h>
#include <Spirit/State.h>
#include <Spirit/System.h>
#include <Spirit/Version.h>
#include <data/State.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Vectormath_Defines.hpp>
#include <utility/Constants.hpp>

#include "catch.hpp"
#include "engine/Neighbours.hpp"
#include "matchers.hpp"
#include "utility.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>

#include <cmath>
#include <map>

// Reduce required precision if float accuracy
#ifdef SPIRIT_SCALAR_TYPE_DOUBLE
[[maybe_unused]] constexpr int digits_a = 12;
[[maybe_unused]] constexpr int digits_b = 4;
[[maybe_unused]] constexpr int digits_c = 1;
#else
[[maybe_unused]] constexpr int digits_a = 1;
[[maybe_unused]] constexpr int digits_b = 4;
[[maybe_unused]] constexpr int digits_c = 1;
#endif

#ifdef SPIRIT_SCALAR_TYPE_DOUBLE
[[maybe_unused]] constexpr scalar epsilon_2 = 1e-5;
[[maybe_unused]] constexpr scalar epsilon_3 = 1e-6;
[[maybe_unused]] constexpr scalar epsilon_4 = 1e-7;
[[maybe_unused]] constexpr scalar epsilon_5 = 1e-8;
[[maybe_unused]] constexpr scalar epsilon_6 = 1e-9;
#else
[[maybe_unused]] constexpr scalar epsilon_2 = 1e-2;
[[maybe_unused]] constexpr scalar epsilon_3 = 1e-3;
[[maybe_unused]] constexpr scalar epsilon_4 = 1e-4;
[[maybe_unused]] constexpr scalar epsilon_5 = 1e-5;
[[maybe_unused]] constexpr scalar epsilon_6 = 1e-6;
#endif

using Catch::CustomMatchers::MapApprox;
using Catch::CustomMatchers::within_digits;
using Catch::Matchers::Equals;
using Catch::Matchers::WithinAbs;

TEST_CASE( "neighbors", "[pairs]" )
{
    SECTION( "crystal structure: HCP" )
    {
        const std::vector<Vector3> basis   = { { 0., 0., 0. }, { 2. / 3., 1. / 3., 0.5 } };
        const std::vector<Vector3> bravais = {
            { 0.5, -0.5 * std::sqrt( 3 ), 0 },
            { 0.5, 0.5 * std::sqrt( 3 ), 0 },
            { 0, 0, 1.633 },
        };
        const auto composition = Data::Basis_Cell_Composition::make_default( basis.size() );
        Data::Geometry geometry( { 0, 0, 0 }, bravais, { 5, 5, 5 }, basis, composition, 1.3, {}, {} );

        const auto shell_radii = Engine::Neighbours::Get_Shell_Radii( geometry, 5 );
        INFO(
            [&shell_radii]
            {
                std::ostringstream oss;
                oss << "shell-radii: ";
                for( const auto & radius : shell_radii )
                    std::cout << radius << " ";
                std::cout << '\n';
                return oss.str();
            }() );

        {
            const std::size_t n_shells = 1;
            pairfield pairs{};
            intfield shells{};

            Engine::Neighbours::Get_Neighbours_in_Shells(
                geometry, n_shells, pairs, shells, /*use_redundant_neighbours=*/true );

            for( unsigned int i = 0; i < pairs.size(); ++i )
                INFO( fmt::format( "shell={}, pair={}\n", shells[i], pairs[i] ) );
            REQUIRE( pairs.size() == shells.size() );
            REQUIRE( pairs.size() == 2 * 12 );
        }

        {
            const std::size_t n_shells = 2;
            pairfield pairs{};
            intfield shells{};

            Engine::Neighbours::Get_Neighbours_in_Shells(
                geometry, n_shells, pairs, shells, /*use_redundant_neighbours=*/true );

            for( unsigned int i = 0; i < pairs.size(); ++i )
                INFO( fmt::format( "shell={}, pair={}\n", shells[i], pairs[i] ) );
            REQUIRE( pairs.size() == shells.size() );
            REQUIRE( pairs.size() == 2 * ( 12 + 6 ) );
        }

        // TODO: list out and compare against the explicit pairs
    }

    // TODO: other crystal structures
}
