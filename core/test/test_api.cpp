#include <Spirit/Chain.h>
#include <Spirit/Configurations.h>
#include <Spirit/Quantities.h>
#include <Spirit/Simulation.h>
#include <Spirit/State.h>
#include <Spirit/System.h>

#include <data/State.hpp>
#include <utility/Exception.hpp>

#include <catch.hpp>

auto inputfile = "core/test/input/api.cfg";
// Reduce required precision if float accuracy
#ifdef SPIRIT_SCALAR_TYPE_DOUBLE
[[maybe_unused]] constexpr scalar epsilon_rough = 1e-12;
#else
[[maybe_unused]] constexpr scalar epsilon_rough = 1e-6;
#endif

using Catch::Matchers::WithinAbs;

TEST_CASE( "State", "[state]" )
{
    SECTION( "State setup, minimal simulation, and state deletion should not throw" )
    {
        std::shared_ptr<State> state;

        // Test the default config explicitly
        CHECK_NOTHROW( state = std::shared_ptr<State>( State_Setup(), State_Delete ) );
        CHECK_NOTHROW( Configuration_PlusZ( state.get() ) );
        CHECK_NOTHROW( Simulation_LLG_Start( state.get(), Solver_VP, 1 ) );

        // Test the default config with a nonexistent file
        CHECK_NOTHROW(
            state
            = std::shared_ptr<State>( State_Setup( "__surely__this__file__does__not__exist__.cfg" ), State_Delete ) );
        CHECK_NOTHROW( Configuration_PlusZ( state.get() ) );
        CHECK_NOTHROW( Simulation_LLG_Start( state.get(), Solver_VP, 1 ) );

        // Test the default input file
        CHECK_NOTHROW( state = std::shared_ptr<State>( State_Setup( inputfile ), State_Delete ) );

        // TODO: actual test
    }

    SECTION( "from_indices()" )
    {
        // Create a state with two images. Let the second one to be the active
        auto state = std::shared_ptr<State>( State_Setup( inputfile ), State_Delete );
        Chain_Image_to_Clipboard( state.get(), 0, 0 );   // Copy to "clipboard"
        Chain_Insert_Image_Before( state.get(), 0, 0 );  // Add before active
        REQUIRE( Chain_Get_NOI( state.get() ) == 2 );    // Number of images is 2
        REQUIRE( System_Get_Index( state.get() ) == 1 ); // Active is 2nd image

        // Arguments for from_indices()
        int idx_image{}, idx_chain{};
        decltype( from_indices( state.get(), idx_image, idx_chain ) ) ret_val;

        // A positive, index beyond the size of the chain should throw an exception
        idx_chain = 0;
        idx_image = 5;
        CHECK_THROWS_AS( ret_val = from_indices( state.get(), idx_image, idx_chain ), Utility::Exception );
        // TODO: find a way to see if the exception thrown was the right one

        // A negative image index should translate to the active image
        idx_chain = 0;
        idx_image = -5;
        CHECK_NOTHROW( ret_val = from_indices( state.get(), idx_image, idx_chain ) );
        REQUIRE( idx_image == 1 );

        // A negative chain index should translate to the active chain
        idx_chain = -5;
        idx_image = 0;
        CHECK_NOTHROW( ret_val = from_indices( state.get(), idx_image, idx_chain ) );
        REQUIRE( idx_chain == 0 );
    }
}

TEST_CASE( "Configurations", "[configurations]" )
{
    auto state = std::shared_ptr<State>( State_Setup( inputfile ), State_Delete );

    // Filters
    scalar position[3]{ 0, 0, 0 };
    scalar r_cut_rectangular[3]{ -1, -1, -1 };
    scalar r_cut_cylindrical = -1;
    scalar r_cut_spherical   = -1;
    bool inverted            = false;

    SECTION( "Domain" )
    {
        scalar dir[3] = { 0, 0, 1 };
        Configuration_PlusZ( state.get(), position, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted );
        // TODO: actual test

        Configuration_MinusZ( state.get(), position, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted );
        // TODO: actual test

        Configuration_Domain(
            state.get(), dir, position, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted );
        // TODO: actual test
    }
    SECTION( "Random" )
    {
        scalar temperature = 5;
        Configuration_Add_Noise_Temperature(
            state.get(), temperature, position, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted );
        // TODO: actual test

        Configuration_Random( state.get(), position, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted );
        // TODO: actual test
    }
    SECTION( "Skyrmion" )
    {
        scalar r     = 5;
        int order    = 1;
        scalar phase = 0;
        bool updown = false, achiral = false, rl = false;
        Configuration_Skyrmion(
            state.get(), r, order, phase, updown, achiral, rl, position, r_cut_rectangular, r_cut_cylindrical,
            r_cut_spherical, inverted );
        // TODO: actual test

        r     = 7;
        order = 1;
        phase = -90, updown = false;
        achiral = true;
        rl      = false;
        Configuration_Skyrmion(
            state.get(), r, order, phase, updown, achiral, rl, position, r_cut_rectangular, r_cut_cylindrical,
            r_cut_spherical, inverted );
        // TODO: actual test
    }
    SECTION( "Hopfion" )
    {
        scalar r         = 5;
        int order        = 1;
        scalar normal[3] = { 0, 0, 1 };
        Configuration_Hopfion(
            state.get(), r, order, position, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted, normal );
        // TODO: actual test
    }
    SECTION( "Spin Spiral" )
    {
        auto dir_type = "real lattice";
        scalar q[3]{ 0, 0, 0.1 }, axis[3]{ 0, 0, 1 }, theta{ 30 };
        Configuration_SpinSpiral(
            state.get(), dir_type, q, axis, theta, position, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical,
            inverted );
        // TODO: actual test
    }
}

TEST_CASE( "Quantities", "[quantities]" )
{
    SECTION( "Magnetization" )
    {
        auto state  = std::shared_ptr<State>( State_Setup( inputfile ), State_Delete );
        scalar m[3] = { 0, 0, 1 };

        SECTION( "001" )
        {
            scalar dir[3] = { 0, 0, 1 };

            Configuration_Domain( state.get(), dir );
            Quantity_Get_Magnetization( state.get(), m );

            REQUIRE_THAT( m[0], WithinAbs( dir[0], 1e-12 ) );
            REQUIRE_THAT( m[1], WithinAbs( dir[1], 1e-12 ) );
            REQUIRE_THAT( m[2], WithinAbs( dir[2], 1e-12 ) );
        }
        SECTION( "011" )
        {
            scalar dir[3] = { 0, 0, 1 };

            Configuration_Domain( state.get(), dir );
            Quantity_Get_Magnetization( state.get(), m );

            REQUIRE_THAT( m[0], WithinAbs( dir[0], 1e-12 ) );
            REQUIRE_THAT( m[1], WithinAbs( dir[1], 1e-12 ) );
            REQUIRE_THAT( m[2], WithinAbs( dir[2], 1e-12 ) );
        }
    }
    SECTION( "Topological Charge" )
    {
        auto state = std::shared_ptr<State>( State_Setup( inputfile ), State_Delete );

        SECTION( "negative charge" )
        {
            Configuration_PlusZ( state.get() );
            Configuration_Skyrmion( state.get(), 6.0, 1.0, -90.0, false, false, false );
            scalar charge = Quantity_Get_Topological_Charge( state.get() );
            REQUIRE_THAT( charge, WithinAbs( -1, epsilon_rough ) );
        }

        SECTION( "positive charge" )
        {
            Configuration_MinusZ( state.get() );
            Configuration_Skyrmion( state.get(), 6.0, 1.0, -90.0, true, false, false );
            scalar charge = Quantity_Get_Topological_Charge( state.get() );
            REQUIRE_THAT( charge, WithinAbs( 1, epsilon_rough ) );
        }
    }
}
