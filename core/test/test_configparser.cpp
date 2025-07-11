#include <Spirit/Configurations.h>
#include <Spirit/Constants.h>
#include <Spirit/Hamiltonian.h>
#include <Spirit/Parameters_EMA.h>
#include <Spirit/Parameters_MMF.h>
#include <Spirit/Simulation.h>
#include <Spirit/State.h>
#include <Spirit/System.h>
#include <Spirit/Version.h>
#include <data/State.hpp>
#include <engine/Vectormath.hpp>

#include "catch.hpp"
#include "matchers.hpp"
#include "utility.hpp"

using Catch::CustomMatchers::MapApprox;
using Catch::Matchers::Equals;
using Catch::Matchers::WithinAbs;

// Reduce required precision if float accuracy
#ifdef SPIRIT_SCALAR_TYPE_DOUBLE
[[maybe_unused]] constexpr scalar epsilon_2 = 1e-10;
[[maybe_unused]] constexpr scalar epsilon_3 = 1e-12;
[[maybe_unused]] constexpr scalar epsilon_4 = 1e-12;
[[maybe_unused]] constexpr scalar epsilon_5 = 1e-6;
[[maybe_unused]] constexpr scalar epsilon_6 = 1e-7;
#else
[[maybe_unused]] constexpr scalar epsilon_2 = 1e-2;
[[maybe_unused]] constexpr scalar epsilon_3 = 1e-3;
[[maybe_unused]] constexpr scalar epsilon_4 = 1e-4;
[[maybe_unused]] constexpr scalar epsilon_5 = 1e-5;
[[maybe_unused]] constexpr scalar epsilon_6 = 1e-6;
#endif

// TODO: Implement proper `Get_{Exchange,DMI}_Shells` API functions and do the same for neighbours.
TEST_CASE( "Hamiltonian: Parse Hamiltonian (Pairs) config and check parsed values using the C-API", "[configparser]" )
{
    static constexpr auto input_file = "core/test/input/configparser_hamiltonian_pairs.toml";

    auto state = std::shared_ptr<State>( State_Setup( input_file ), State_Delete );
    REQUIRE( state != nullptr );
    REQUIRE( !state->config_file.empty() );

    SECTION( "External field" )
    {
        scalar magnitude_ref{ 25.0 };
        const scalar component_ref = 1.0 / std::sqrt( 2.0 );
        std::array<scalar, 3> direction_ref{ 0.0, component_ref, component_ref };

        scalar magnitude{};
        std::array<scalar, 3> direction{};

        Hamiltonian_Get_Field( state.get(), &magnitude, direction.data() );

        REQUIRE_THAT( magnitude, WithinAbs( magnitude_ref, epsilon_2 ) );
        for( auto i = 0; i < 3; ++i )
            REQUIRE_THAT( direction[i], WithinAbs( direction_ref[i], epsilon_2 ) );
    }

    SECTION( "Uniaxial Anisotropy" )
    {
        constexpr scalar magnitude_ref{ 1.0 };
        constexpr std::array<scalar, 3> direction_ref{ 0.0, 0.0, 1.0 };

        scalar magnitude{};
        std::array<scalar, 3> direction{};

        Hamiltonian_Get_Anisotropy( state.get(), &magnitude, direction.data() );

        REQUIRE_THAT( magnitude, WithinAbs( magnitude_ref, epsilon_2 ) );
        for( auto i = 0; i < 3; ++i )
            REQUIRE_THAT( direction[i], WithinAbs( direction_ref[i], epsilon_2 ) );
    }

    SECTION( "Cubic Anisotropy" )
    {
        constexpr scalar magnitude_ref{ 4.0 };
        scalar magnitude{};

        Hamiltonian_Get_Cubic_Anisotropy( state.get(), &magnitude );

        REQUIRE_THAT( magnitude, WithinAbs( magnitude_ref, epsilon_2 ) );
    }

    SECTION( "Biaxial Anisotropy" )
    {
        constexpr int n_atoms_ref        = 1;
        constexpr int n_terms_ref        = 7;
        constexpr std::array indices_ref = { 0 };
        constexpr std::array<std::array<scalar, 3>, 1> primary_ref{ { { 0, 0, 1 } } };
        constexpr std::array<std::array<scalar, 3>, 1> secondary_ref{ { { 1, 0, 0 } } };
        constexpr std::array site_p_ref{ 0, 7 };
        constexpr std::array<scalar, 7> magnitudes_ref{ 3.0, 1.8, 0.9, -3.2, 3.2, -1.6, 1.6 };
        constexpr std::array exponents_ref{
            std::array{ 1, 0, 0 }, std::array{ 2, 0, 0 }, std::array{ 3, 0, 0 }, std::array{ 1, 2, 0 },
            std::array{ 0, 4, 0 }, std::array{ 3, 2, 0 }, std::array{ 3, 4, 0 },
        };

        const int n_atoms = Hamiltonian_Get_Biaxial_Anisotropy_N_Atoms( state.get() );

        REQUIRE( n_atoms > 0 );
        REQUIRE( n_atoms == n_atoms_ref );

        const int n_terms = Hamiltonian_Get_Biaxial_Anisotropy_N_Terms( state.get() );

        REQUIRE( n_terms > n_atoms );
        REQUIRE( n_terms == n_terms_ref );

        std::vector<int> indices( n_atoms );
        std::vector<std::array<scalar, 3>> primary( n_atoms );
        std::vector<std::array<scalar, 3>> secondary( n_atoms );
        std::vector<int> site_p( n_atoms + 1 );
        std::vector<scalar> magnitudes( n_terms );
        std::vector exponents( n_terms, std::array{ 0, 0, 0 } );

        Hamiltonian_Get_Biaxial_Anisotropy(
            state.get(), indices.data(), array_cast( primary ), array_cast( secondary ), site_p.data(), n_atoms,
            magnitudes.data(), array_cast( exponents ), n_terms );

        if( n_atoms > 0 )
        {
            REQUIRE( site_p[0] == 0 );
            REQUIRE( site_p.size() == static_cast<unsigned int>( n_atoms ) + 1 );
        }

        for( int i = 0; i < n_atoms; ++i )
            REQUIRE( site_p[i] < site_p[i + 1] );

        for( std::size_t i = 0; i < site_p.size(); ++i )
            REQUIRE( site_p[i] == site_p_ref[i] );

        for( std::size_t i = 0; i < indices.size(); ++i )
            REQUIRE( indices[i] == indices_ref[i] );

        for( int i = 0; i < n_atoms; ++i )
        {
#pragma unroll
            for( int j = 0; j < 3; ++j )
            {
                REQUIRE_THAT( primary[i][j], WithinAbs( primary_ref[i][j], epsilon_2 ) );
                REQUIRE_THAT( secondary[i][j], WithinAbs( secondary_ref[i][j], epsilon_2 ) );
            }
        }

        // use std::map to compare them, because the order of the polynomial terms need not be fixed
        using term_idx = std::tuple<int, int, int>;
        using term_map = std::map<term_idx, scalar>;
        auto make_polynomial
            = []( const int offset_begin, const int offset_end, const auto & exponents, const auto & magnitudes )
        {
            term_map polynomial{};
            for( int i = offset_begin; i < offset_end; ++i )
                polynomial.emplace(
                    std::make_tuple( exponents[i][0], exponents[i][1], exponents[i][2] ), magnitudes[i] );
            return polynomial;
        };

        for( std::size_t i = 1; i < site_p.size(); ++i )
        {
            const auto polynomial_ref
                = make_polynomial( site_p_ref[i - 1], site_p_ref[i], exponents_ref, magnitudes_ref );
            const auto polynomial = make_polynomial( site_p[i - 1], site_p[i], exponents, magnitudes );

            REQUIRE_THAT( polynomial, MapApprox( polynomial_ref, epsilon_2 ) );
        }
    }

    SECTION( "Heisenberg Exchange" )
    {
        constexpr std::array<std::array<int, 2>, 3> indices_ref
            = { std::array{ 0, 0 }, std::array{ 0, 0 }, std::array{ 0, 0 } };
        constexpr std::array<std::array<int, 3>, 3> translations_ref
            = { std::array{ 1, 0, 0 }, std::array{ 0, 1, 0 }, std::array{ 0, 0, 1 } };
        constexpr std::array<scalar, 3> Jij_ref{ 10.0, 10.0, 10.0 };

        const int n_pairs = Hamiltonian_Get_Exchange_N_Pairs( state.get() );

        std::vector<std::array<int, 2>> indices( n_pairs );
        std::vector<std::array<int, 3>> translations( n_pairs );
        std::vector<scalar> Jij( n_pairs );

        Hamiltonian_Get_Exchange_Pairs( state.get(), array_cast( indices ), array_cast( translations ), Jij.data() );

        // use std::map to compare them, because the order of the exchange terms need not be fixed
        using exchange_idx = std::tuple<int, int, int, int, int>;
        using exchange_map = std::map<exchange_idx, scalar>;
        exchange_map terms, terms_ref;
        REQUIRE( n_pairs > 0 );

        auto make_exchange_map = []( const auto & indices, const auto & translations, const auto & Jij )
        {
            exchange_map terms{};
            for( std::size_t i = 0; i < indices.size(); ++i )
                terms.emplace(
                    std::make_tuple(
                        indices[i][0], indices[i][1], translations[i][0], translations[i][1], translations[i][2] ),
                    Jij[i] );
            return terms;
        };

        const auto exchange     = make_exchange_map( indices, translations, Jij );
        const auto exchange_ref = make_exchange_map( indices_ref, translations_ref, Jij_ref );

        REQUIRE_THAT( exchange, MapApprox( exchange_ref, epsilon_2 ) );
    }

    SECTION( "DMI" )
    {
        // TODO: DMI once the API supports it
        const int n_pairs = Hamiltonian_Get_DMI_N_Pairs( state.get() );
    }

    SECTION( "Dipole-Dipole Interaction" )
    {
        constexpr int ddi_method_ref = static_cast<int>( Engine::Spin::DDI_Method::None );
        constexpr std::array<int, 3> n_periodic_images_ref{ 4, 4, 4 };
        constexpr scalar cutoff_radius_ref = 1.3;
        constexpr bool pb_zero_padding_ref = true;

        int ddi_method = 0;
        std::array<int, 3> n_periodic_images{};
        scalar cutoff_radius = 0;
        bool pb_zero_padding = false;

        Hamiltonian_Get_DDI( state.get(), &ddi_method, n_periodic_images.data(), &cutoff_radius, &pb_zero_padding );

        REQUIRE( ddi_method == ddi_method_ref );
        REQUIRE( n_periodic_images == n_periodic_images_ref );
        REQUIRE_THAT( cutoff_radius, WithinAbs( cutoff_radius_ref, epsilon_2 ) );
        REQUIRE( pb_zero_padding == pb_zero_padding_ref );
    }
}
TEST_CASE( "Geometry", "[configparser]" )
{
    static constexpr auto input_file = "core/test/input/configparser_geometry.toml";

    // Create a state with two images. Let the second one to be the active
    auto state = std::shared_ptr<State>( State_Setup( input_file ), State_Delete );
    REQUIRE( state != nullptr );
    REQUIRE( !state->config_file.empty() );

    SECTION( "Basis" )
    {
        const auto ref_basis = std::array<Vector3, 2>{ {
            { 0, 0, 0 },
            { 0.5, 0.5, 0 },
        } };

        REQUIRE( ref_basis.size() == Geometry_Get_N_Cell_Atoms( state.get() ) );

        Vector3 * basis     = nullptr;
        std::size_t n_basis = Geometry_Get_Cell_Atoms( state.get(), reinterpret_cast<scalar **>( &basis ) );
        REQUIRE( n_basis == ref_basis.size() );

        for( unsigned int i = 0; i < ref_basis.size() && i < n_basis; ++i )
        {
            const auto & ref   = ref_basis[i];
            const auto & found = basis[i];
            INFO( fmt::format( "basis vector (expected): {} {} {}", ref[0], ref[1], ref[2] ) );
            INFO( fmt::format( "basis vector (found):    {} {} {}", found[0], found[1], found[2] ) )
            REQUIRE_THAT( ( ref_basis[i] - basis[i] ).norm(), WithinAbs( 0, epsilon_2 ) );
        }
    }
}

// TODO: Add verification for parsing any of the parameter sets.
TEST_CASE( "Parameters LLG: Parse config and check parsed values using the C-API", "[configparser]" )
{
    static constexpr auto input_file = "core/test/input/configparser_parameters_llg.toml";

    auto state = std::shared_ptr<State>( State_Setup( input_file ), State_Delete );
    REQUIRE( state != nullptr );
    REQUIRE( !state->config_file.empty() );

    // TODO: make sure that the provided values aren't the default ones,
    // Otherwise the test also passes when parsing fails completely.
    const long ref_max_walltime     = 0;
    const int ref_seed              = 20006;
    const int n_iterations          = 2000000;
    const int n_iterations_log      = 2000;
    const int n_iterations_amortize = 1;

    const scalar ref_temperature                     = 42;
    const scalar ref_temperature_gradient_magnitude  = 3;
    const Vector3 ref_temperature_gradient_direction = { 0, 1, 0 };

    const scalar damping     = 0.7;
    const scalar beta        = 0.3;
    const scalar dt          = 1.0e-3;
    const bool renorm        = false;
    const scalar convergence = 10e-9;

    const bool output_any     = true;
    const bool output_initial = true;
    const bool output_final   = true;

    const bool output_energy_step                  = false;
    const bool output_energy_archive               = true;
    const bool output_energy_spin_resolved         = false;
    const bool output_energy_divide_by_nspins      = true;
    const bool output_energy_add_readability_lines = true;

    const bool output_configuration_step    = true;
    const bool output_configuration_archive = false;
    const int output_configuration_filetype = 3;

    const auto spin_current_model               = Data::SC_Model::TRANSFER_TORQUE;
    const scalar spin_current_vector_magnitude  = 1.0;
    const Vector3 spin_current_vector_direction = { 0.0, 1.0, 0.0 };

    SECTION( "Simulation: Iterations" )
    {
        int iterations, iterations_log;
        Parameters_LLG_Get_N_Iterations( state.get(), &iterations, &iterations_log );

        REQUIRE( iterations == n_iterations );
        REQUIRE( iterations_log == n_iterations_log );
    }
    SECTION( "Simulation: General" )
    {
        REQUIRE_THAT( Parameters_LLG_Get_Damping( state.get() ), WithinAbs( damping, epsilon_2 ) );
        REQUIRE_THAT( Parameters_LLG_Get_Non_Adiabatic_Damping( state.get() ), WithinAbs( beta, epsilon_2 ) );
        REQUIRE_THAT( Parameters_LLG_Get_Time_Step( state.get() ), WithinAbs( dt, epsilon_2 ) );
        REQUIRE_THAT( Parameters_LLG_Get_Convergence( state.get() ), WithinAbs( convergence, epsilon_2 ) );
    }
    SECTION( "Simulation: Temperature" )
    {
        REQUIRE_THAT( Parameters_LLG_Get_Temperature( state.get() ), WithinAbs( ref_temperature, epsilon_2 ) );

        scalar magnitude;
        Vector3 direction = Vector3::Zero();
        Parameters_LLG_Get_Temperature_Gradient( state.get(), &magnitude, direction.data() );

        REQUIRE_THAT( magnitude, WithinAbs( ref_temperature_gradient_magnitude, epsilon_2 ) );
        REQUIRE( direction.isApprox( ref_temperature_gradient_direction, epsilon_2 ) );
    }
    SECTION( "Simulation: Spin Current" )
    {
        int model         = 0;
        scalar magnitude  = 0;
        Vector3 direction = Vector3::Zero();

        Parameters_LLG_Get_Spin_Current( state.get(), &model, &magnitude, direction.data() );
        REQUIRE( static_cast<Data::SC_Model>( model ) == spin_current_model );
        REQUIRE_THAT( magnitude, WithinAbs( spin_current_vector_magnitude, epsilon_2 ) );
        REQUIRE( direction.isApprox( spin_current_vector_direction, epsilon_2 ) );
    }
    SECTION( "Output: General" )
    {
        bool any, initial, final;
        Parameters_LLG_Get_Output_General( state.get(), &any, &initial, &final );
        REQUIRE( any == output_any );
        REQUIRE( initial == output_initial );
        REQUIRE( final == output_final );
    }
    SECTION( "Output: Energy" )
    {
        bool step, archive, spin_resolved, divide_by_nspins, readability_lines;
        Parameters_LLG_Get_Output_Energy(
            state.get(), &step, &archive, &spin_resolved, &divide_by_nspins, &readability_lines );
        REQUIRE( step == output_energy_step );
        REQUIRE( archive == output_energy_archive );
        REQUIRE( spin_resolved == output_energy_spin_resolved );
        REQUIRE( divide_by_nspins == output_energy_divide_by_nspins );
        REQUIRE( readability_lines == output_energy_add_readability_lines );
    }
    SECTION( "Output: Configuration" )
    {
        bool step, archive;
        int filetype;
        Parameters_LLG_Get_Output_Configuration( state.get(), &step, &archive, &filetype );
        REQUIRE( step == output_configuration_step );
        REQUIRE( archive == output_configuration_archive );
        REQUIRE( filetype == output_configuration_filetype );
    }
    SECTION( "Output: Path" )
    {
        REQUIRE_THAT( Parameters_LLG_Get_Output_Tag( state.get() ), Equals( "test_configparser_llg" ) );
        REQUIRE_THAT( Parameters_LLG_Get_Output_Folder( state.get() ), Equals( "output" ) );
    }
}

TEST_CASE( "Parameters MC: Parse config and check parsed values using the C-API", "[configparser]" )
{
    static constexpr auto input_file = "core/test/input/configparser_parameters_mc.toml";

    auto state = std::shared_ptr<State>( State_Setup( input_file ), State_Delete );
    REQUIRE( state != nullptr );
    REQUIRE( !state->config_file.empty() );

    // TODO: make sure that the provided values aren't the default ones,
    // Otherwise the test also passes when parsing fails completely.
    const long ref_max_walltime     = 0;
    const int ref_seed              = 20006;
    const int n_iterations          = 2000000;
    const int n_iterations_log      = 2000;
    const int n_iterations_amortize = 1;

    const scalar ref_temperature = 42;

    const int metropolis_step                       = MC_Metropolis_Step_Spin_Sphere;
    const bool metropolis_use_adaptive_cone         = false;
    const bool metropolis_random_sample             = false;
    const scalar metropolis_cone_angle              = 25.0;
    const scalar metropolis_target_acceptance_ratio = 0.3;

    const bool output_any     = true;
    const bool output_initial = true;
    const bool output_final   = true;

    const bool output_energy_step                  = false;
    const bool output_energy_archive               = true;
    const bool output_energy_spin_resolved         = false;
    const bool output_energy_divide_by_nspins      = true;
    const bool output_energy_add_readability_lines = true;

    const bool output_configuration_step    = true;
    const bool output_configuration_archive = false;
    const int output_configuration_filetype = 3;

    SECTION( "Simulation: Iterations" )
    {
        int iterations, iterations_log;
        Parameters_MC_Get_N_Iterations( state.get(), &iterations, &iterations_log );

        REQUIRE( iterations == n_iterations );
        REQUIRE( iterations_log == n_iterations_log );
    }
    SECTION( "Simulation: General" )
    {
        REQUIRE_THAT( Parameters_MC_Get_Temperature( state.get() ), WithinAbs( ref_temperature, epsilon_2 ) );

        Parameters_MC_Metropolis_Parameters params;
        Parameters_MC_Get_Metropolis_Parameters( state.get(), &params );

        REQUIRE( params.step == metropolis_step );
        REQUIRE( params.use_adaptive_cone == metropolis_use_adaptive_cone );
        REQUIRE_THAT( params.cone_angle, WithinAbs( metropolis_cone_angle, epsilon_2 ) );
        REQUIRE_THAT( params.target_acceptance_ratio, WithinAbs( metropolis_target_acceptance_ratio, epsilon_2 ) );
    }
    SECTION( "Output: General" )
    {
        bool any, initial, final;
        Parameters_MC_Get_Output_General( state.get(), &any, &initial, &final );
        REQUIRE( any == output_any );
        REQUIRE( initial == output_initial );
        REQUIRE( final == output_final );
    }
    SECTION( "Output: Energy" )
    {
        bool step, archive, spin_resolved, divide_by_nspins, readability_lines;
        Parameters_MC_Get_Output_Energy(
            state.get(), &step, &archive, &spin_resolved, &divide_by_nspins, &readability_lines );
        REQUIRE( step == output_energy_step );
        REQUIRE( archive == output_energy_archive );
        REQUIRE( spin_resolved == output_energy_spin_resolved );
        REQUIRE( divide_by_nspins == output_energy_divide_by_nspins );
        REQUIRE( readability_lines == output_energy_add_readability_lines );
    }
    SECTION( "Output: Configuration" )
    {
        bool step, archive;
        int filetype;
        Parameters_MC_Get_Output_Configuration( state.get(), &step, &archive, &filetype );
        REQUIRE( step == output_configuration_step );
        REQUIRE( archive == output_configuration_archive );
        REQUIRE( filetype == output_configuration_filetype );
    }
    SECTION( "Output: Path" )
    {
        REQUIRE_THAT( Parameters_MC_Get_Output_Tag( state.get() ), Equals( "test_configparser_mc" ) );
        REQUIRE_THAT( Parameters_MC_Get_Output_Folder( state.get() ), Equals( "output" ) );
    }
}

TEST_CASE( "Parameters GNEB: Parse config and check parsed values using the C-API", "[configparser]" )
{
    static constexpr auto input_file = "core/test/input/configparser_parameters_gneb.toml";

    auto state = std::shared_ptr<State>( State_Setup( input_file ), State_Delete );
    REQUIRE( state != nullptr );
    REQUIRE( !state->config_file.empty() );

    const long int max_walltime_sec      = 0;
    const int rng_seed                   = 2006;
    const long int n_iterations          = 20000;
    const long int n_iterations_log      = 20;
    const long int n_iterations_amortize = 1;

    const scalar temperature = 0;

    const double force_convergence        = 1e-7;
    const scalar spring_constant          = 1.5;
    const scalar spring_force_ratio       = 0; // Not part of the config
    const scalar path_shortening_constant = 0; // Not part of the config
    const int n_E_interpolations          = 10;

    const bool moving_endpoints      = true;
    const bool translating_endpoints = true;

    const scalar equilibrium_delta_Rx_left  = 0.5;
    const scalar equilibrium_delta_Rx_right = 0.5;

    const bool escape_first = false; // Neither part of the config nor the API

    const std::string output_folder   = "output";
    const std::string output_file_tag = "<time>";
    const bool output_any             = true;
    const bool output_initial         = false;
    const bool output_final           = false;

    const bool output_energies_step                  = false;
    const bool output_energies_divide_by_nspins      = true;
    const bool output_energies_add_readability_lines = true;
    const bool output_energies_interpolated          = true;
    const bool output_chain_step                     = true;
    const int output_vf_filetype                     = 3;

    SECTION( "Simulation: Iterations" )
    {
        int iterations, iterations_log;
        Parameters_GNEB_Get_N_Iterations( state.get(), &iterations, &iterations_log );

        REQUIRE( iterations == n_iterations );
        REQUIRE( iterations_log == n_iterations_log );
    }
    SECTION( "Simulation: General" )
    {
        REQUIRE( moving_endpoints == Parameters_GNEB_Get_Moving_Endpoints( state.get() ) );
        REQUIRE( translating_endpoints == Parameters_GNEB_Get_Translating_Endpoints( state.get() ) );
        REQUIRE( n_E_interpolations == Parameters_GNEB_Get_N_Energy_Interpolations( state.get() ) );

        scalar left, right;
        Parameters_GNEB_Get_Equilibrium_Delta_Rx( state.get(), &left, &right );
        REQUIRE_THAT( left, WithinAbs( equilibrium_delta_Rx_left, epsilon_2 ) );
        REQUIRE_THAT( right, WithinAbs( equilibrium_delta_Rx_right, epsilon_2 ) );

        REQUIRE_THAT( Parameters_GNEB_Get_Convergence( state.get() ), WithinAbs( force_convergence, epsilon_2 ) );
        REQUIRE_THAT( Parameters_GNEB_Get_Spring_Constant( state.get() ), WithinAbs( spring_constant, epsilon_2 ) );

        // REQUIRE_THAT(
        //     Parameters_GNEB_Get_Spring_Force_Ratio( state.get() ), WithinAbs( spring_force_ratio, epsilon_2 ) );
        // REQUIRE_THAT(
        //     Parameters_GNEB_Get_Path_Shortening_Constant( state.get() ),
        //     WithinAbs( path_shortening_constant, epsilon_2 ) );
    }
    SECTION( "Simulation: Temperature" ) {}
    SECTION( "Output: General" )
    {
        bool any, initial, final;
        Parameters_GNEB_Get_Output_General( state.get(), &any, &initial, &final );
        REQUIRE( any == output_any );
        REQUIRE( initial == output_initial );
        REQUIRE( final == output_final );
    }
    SECTION( "Output: Energies" )
    {
        bool step, interpolated, divide_by_nspins, readability_lines;
        Parameters_GNEB_Get_Output_Energies( state.get(), &step, &interpolated, &divide_by_nspins, &readability_lines );
        REQUIRE( step == output_energies_step );
        REQUIRE( interpolated == output_energies_interpolated );
        REQUIRE( divide_by_nspins == output_energies_divide_by_nspins );
        REQUIRE( readability_lines == output_energies_add_readability_lines );
    }
    SECTION( "Output: Configuration" )
    {
        bool step;
        int filetype;
        Parameters_GNEB_Get_Output_Chain( state.get(), &step, &filetype );
        REQUIRE( step == output_chain_step );
        REQUIRE( filetype == output_vf_filetype );
    }
    SECTION( "Output: Path" )
    {
        REQUIRE_THAT( Parameters_GNEB_Get_Output_Tag( state.get() ), Equals( "test_configparser_gneb" ) );
        REQUIRE_THAT( Parameters_GNEB_Get_Output_Folder( state.get() ), Equals( "output" ) );
    }
}

TEST_CASE( "Parameters EMA: Parse config and check parsed values using the C-API", "[configparser]" )
{
    static constexpr auto input_file = "core/test/input/configparser_parameters_ema.toml";

    auto state = std::shared_ptr<State>( State_Setup( input_file ), State_Delete );
    REQUIRE( state != nullptr );
    REQUIRE( !state->config_file.empty() );

    const int max_walltime_sec           = 0;
    const long int n_iterations          = 20000;
    const long int n_iterations_log      = 20;
    const long int n_iterations_amortize = 1;

    const int n_modes       = 8;
    const int n_mode_follow = 2;

    const scalar frequency = 3.14;
    const scalar amplitude = 4.2;
    const bool snapshot    = false; // not part of the config
    const bool sparse      = true;

    const bool output_any     = true;
    const bool output_initial = false;
    const bool output_final   = false;

    const bool output_energy_step                  = false;
    const bool output_energy_archive               = true;
    const bool output_energy_spin_resolved         = true;
    const bool output_energy_divide_by_nspins      = true;
    const bool output_energy_add_readability_lines = true;
    const bool output_energy_interpolated          = true;
    const int output_vf_filetype                   = 3;

    SECTION( "Simulation: Iterations" )
    {
        // TODO: Not currently part of the API
    }

    SECTION( "Simulation: General" )
    {
        REQUIRE( n_modes == Parameters_EMA_Get_N_Modes( state.get() ) );
        REQUIRE( n_mode_follow == Parameters_EMA_Get_N_Mode_Follow( state.get() ) );
        REQUIRE_THAT( Parameters_EMA_Get_Frequency( state.get() ), WithinAbs( frequency, epsilon_2 ) );
        REQUIRE_THAT( Parameters_EMA_Get_Amplitude( state.get() ), WithinAbs( amplitude, epsilon_2 ) );
        REQUIRE( snapshot == Parameters_EMA_Get_Snapshot( state.get() ) );
        REQUIRE( sparse == Parameters_EMA_Get_Sparse( state.get() ) );
    }

    SECTION( "Output" )
    {
        // TODO: Not currently part of the API
    }
}

TEST_CASE( "Parameters MMF: Parse config and check parsed values using the C-API", "[configparser]" )
{
    static constexpr auto input_file = "core/test/input/configparser_parameters_mmf.toml";

    auto state = std::shared_ptr<State>( State_Setup( input_file ), State_Delete );
    REQUIRE( state != nullptr );
    REQUIRE( !state->config_file.empty() );

    const int max_walltime_sec           = 0;
    const long int n_iterations          = 20000;
    const long int n_iterations_log      = 20;
    const long int n_iterations_amortize = 1;

    const scalar force_convergence = 1e-7; // Not part of the API
    const int n_modes              = 8;
    const int n_mode_follow        = 2;

    const bool output_any     = true;
    const bool output_initial = false;
    const bool output_final   = false;

    const bool output_energy_step                  = false;
    const bool output_energy_archive               = true;
    const bool output_energy_spin_resolved         = true;
    const bool output_energy_divide_by_nspins      = true;
    const bool output_energy_add_readability_lines = true;

    const bool output_configuration_step    = false;
    const bool output_configuration_archive = true;
    const int output_vf_filetype            = 3;

    SECTION( "Simulation: Iterations" )
    {
        int iterations, iterations_log;
        Parameters_MMF_Get_N_Iterations( state.get(), &iterations, &iterations_log );

        REQUIRE( iterations == n_iterations );
        REQUIRE( iterations_log == n_iterations_log );
    }
    SECTION( "Simulation: General" )
    {
        REQUIRE( n_modes == Parameters_MMF_Get_N_Modes( state.get() ) );
        REQUIRE( n_mode_follow == Parameters_MMF_Get_N_Mode_Follow( state.get() ) );
    }
    SECTION( "Output: General" )
    {
        bool any, initial, final;
        Parameters_MMF_Get_Output_General( state.get(), &any, &initial, &final );
        REQUIRE( any == output_any );
        REQUIRE( initial == output_initial );
        REQUIRE( final == output_final );
    }
    SECTION( "Output: Energy" )
    {
        bool step, archive, spin_resolved, divide_by_nspins, readability_lines;
        Parameters_MMF_Get_Output_Energy(
            state.get(), &step, &archive, &spin_resolved, &divide_by_nspins, &readability_lines );
        REQUIRE( step == output_energy_step );
        REQUIRE( archive == output_energy_archive );
        REQUIRE( spin_resolved == output_energy_spin_resolved );
        REQUIRE( divide_by_nspins == output_energy_divide_by_nspins );
        REQUIRE( readability_lines == output_energy_add_readability_lines );
    }
    SECTION( "Output: Configuration" )
    {
        bool step, archive;
        int filetype;
        Parameters_MMF_Get_Output_Configuration( state.get(), &step, &archive, &filetype );
        REQUIRE( step == output_configuration_step );
        REQUIRE( archive == output_configuration_archive );
        REQUIRE( filetype == output_vf_filetype );
    }
    SECTION( "Output: Path" )
    {
        REQUIRE_THAT( Parameters_MMF_Get_Output_Tag( state.get() ), Equals( "test_configparser_mmf" ) );
        REQUIRE_THAT( Parameters_MMF_Get_Output_Folder( state.get() ), Equals( "output" ) );
    }
}
