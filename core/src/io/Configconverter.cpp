#include <io/Configconverter.hpp>
#include <io/Configparser.hpp>
#include <io/Configwriter.hpp>
#include <io/Dataparser.hpp>
#include <io/Filter_File_Handle.hpp>
#include <utility/Logging.hpp>

#include <toml++/toml.hpp>

#include <string>

namespace IO
{

namespace convert
{

using Utility::Log_Level;
using Utility::Log_Sender;

namespace detail
{

auto Defaults( const std::string & config_file_name ) -> IO::Defaults
{
    std::string output_folder = ".";
    std::string file_tag      = "";

    IO::Defaults defaults{};
    try
    {
        if( !config_file_name.empty() )
        {
            try
            {
                Log( Log_Level::Debug, Log_Sender::IO, "Building Defaults" );
                IO::Filter_File_Handle config_file_handle( config_file_name );
                config_file_handle.Read_Single( file_tag, "output_file_tag" );
                if( file_tag != "" )
                    defaults.output.file_tag.emplace( file_tag );

                std::array<std::string, 6> directory_keys{ "ema_output_folder", "gneb_output_folder",
                                                           "llg_output_folder", "mc_output_folder",
                                                           "mmf_output_folder", "log_output_folder" };

                std::unordered_map<std::string, int> counts;
                for( const auto & key : directory_keys )
                {
                    std::string value;
                    config_file_handle.Read_Single( value, key );
                    counts[value] += 1;
                }
                // select default by majority vote
                const auto it = std::max_element(
                    counts.begin(), counts.end(),
                    []( const auto & lhs, const auto & rhs ) { return lhs.second < rhs.second; } );
                if( it != counts.end() && it->second > 1 )
                {
                    defaults.output.directory.emplace( it->first );
                };
            }
            catch( ... )
            {
                spirit_rethrow( fmt::format( "Failed to read default values from file \"{}\".", config_file_name ) );
            }
        }
    }
    catch( ... )
    {
        spirit_handle_exception_core(
            fmt::format( "Unable to read logging parameters from config file \"{}\"", config_file_name ) );
    }
    return defaults;
}

void Apply_Defaults( const IO::Defaults & defaults, toml::table & values )
{
    auto & output = *values["output"].as_table();
    if( auto v = output["file_tag"].as<std::string>(); v && *v == defaults.output.file_tag )
        output.erase( "file_tag" );

    if( auto v = output["folder"].as<std::string>(); v && *v == defaults.output.directory )
        output.erase( "folder" );

    if( output.empty() )
        values.erase( "output" );
}

auto Logging( const std::string & config_file_name, const IO::Defaults & defaults ) -> toml::table
{
    // Verbosity and Reject Level are read as integers
    int i_level_file = 5, i_level_console = 5;
    std::string output_folder = ".";
    std::string file_tag      = "";
    bool messages_to_file = true, messages_to_console = true, save_input_initial = false, save_input_final = false,
         save_positions_initial = false, save_positions_final = false, save_neighbours_initial = false,
         save_neighbours_final = false;
    try
    {
        if( !config_file_name.empty() )
        {
            try
            {
                Log( Log_Level::Debug, Log_Sender::IO, "Building Log" );
                IO::Filter_File_Handle config_file_handle( config_file_name );

                // Time tag
                config_file_handle.Read_Single( file_tag, "output_file_tag" );

                // Output folder
                config_file_handle.Read_Single( output_folder, "log_output_folder" );

                // Save Output (Log Messages) to file
                config_file_handle.Read_Single( messages_to_file, "log_to_file" );
                // File Accept Level
                config_file_handle.Read_Single( i_level_file, "log_file_level" );

                // Print Output (Log Messages) to console
                config_file_handle.Read_Single( messages_to_console, "log_to_console" );
                // File Accept Level
                config_file_handle.Read_Single( i_level_console, "log_console_level" );

                // Save Input (parameters from config file and defaults) on State Setup
                config_file_handle.Read_Single( save_input_initial, "save_input_initial" );
                // Save Input (parameters from config file and defaults) on State Delete
                config_file_handle.Read_Single( save_input_final, "save_input_final" );

                // Save Input (parameters from config file and defaults) on State Setup
                config_file_handle.Read_Single( save_positions_initial, "save_positions_initial" );
                // Save Input (parameters from config file and defaults) on State Delete
                config_file_handle.Read_Single( save_positions_final, "save_positions_final" );

                // Save Input (parameters from config file and defaults) on State Setup
                config_file_handle.Read_Single( save_neighbours_initial, "save_neighbours_initial" );
                // Save Input (parameters from config file and defaults) on State Delete
                config_file_handle.Read_Single( save_neighbours_final, "save_neighbours_final" );
            }
            catch( ... )
            {
                spirit_rethrow( fmt::format(
                    "Failed to read log levels from file \"{}\". Leaving values at default.", config_file_name ) );
            }
        }
    }
    catch( ... )
    {
        spirit_handle_exception_core(
            fmt::format( "Unable to read logging parameters from config file \"{}\"", config_file_name ) );
    }

    auto result = toml::table{
        { "output",
          toml::table{
              { "file_tag", file_tag },
              { "folder", output_folder },
          } },
        { "log_file_level", i_level_file },
        { "log_console_level", i_level_console },
        { "log_to_file", messages_to_file },
        { "log_to_console", messages_to_console },
        { "save_input_initial", save_input_initial },
        { "save_input_final", save_input_final },
        { "save_positions_initial", save_positions_initial },
        { "save_positions_final", save_positions_final },
        { "save_neighbours_initial", save_neighbours_initial },
        { "save_neighbours_final", save_neighbours_final },
    };
    Apply_Defaults( defaults, result );
    return result;
}

auto Parameters_Method_EMA( const std::string & config_file_name, const IO::Defaults & defaults ) -> toml::table
{ // Default parameters
    auto parameters = Data::Parameters_Method_EMA{};

    // Maximum wall time
    std::string str_max_walltime = "0";

    // Parse
    if( !config_file_name.empty() )
    {
        try
        {
            IO::Filter_File_Handle config_file_handle( config_file_name );

            // Output parameters
            config_file_handle.Read_Single( parameters.output_folder, "ema_output_folder" );
            config_file_handle.Read_Single( parameters.output_file_tag, "output_file_tag" );
            config_file_handle.Read_Single( parameters.output_any, "ema_output_any" );
            config_file_handle.Read_Single( parameters.output_initial, "ema_output_initial" );
            config_file_handle.Read_Single( parameters.output_final, "ema_output_final" );
            config_file_handle.Read_Single(
                parameters.output_energy_divide_by_nspins, "ema_output_energy_divide_by_nspins" );
            config_file_handle.Read_Single( parameters.output_energy_spin_resolved, "ema_output_energy_spin_resolved" );
            config_file_handle.Read_Single( parameters.output_energy_step, "ema_output_energy_step" );
            config_file_handle.Read_Single( parameters.output_energy_archive, "ema_output_energy_archive" );
            config_file_handle.Read_Single( parameters.output_configuration_step, "ema_output_configuration_step" );
            config_file_handle.Read_Single(
                parameters.output_configuration_archive, "ema_output_configuration_archive" );
            // Method parameters
            config_file_handle.Read_Single( str_max_walltime, "ema_max_walltime" );
            config_file_handle.Read_Single( parameters.n_iterations, "ema_n_iterations" );
            config_file_handle.Read_Single( parameters.n_iterations_log, "ema_n_iterations_log" );
            config_file_handle.Read_Single( parameters.n_modes, "ema_n_modes" );
            config_file_handle.Read_Single( parameters.n_mode_follow, "ema_n_mode_follow" );
            config_file_handle.Read_Single( parameters.frequency, "ema_frequency" );
            config_file_handle.Read_Single( parameters.amplitude, "ema_amplitude" );
            config_file_handle.Read_Single( parameters.sparse, "ema_sparse" );
        }
        catch( ... )
        {
            spirit_handle_exception_core(
                fmt::format( "Unable to parse EMA parameters from config file \"{}\"", config_file_name ) );
        }
    }
    else
        Log( Log_Level::Parameter, Log_Sender::IO, "Parameters EMA: Using default configuration!" );

    auto result = Parameters_Method_EMA_to_TOML( parameters );
    Apply_Defaults( defaults, result );
    return result;
}; // namespace IO

auto Parameters_Method_GNEB( const std::string & config_file_name, const IO::Defaults & defaults ) -> toml::table
{
    // Default parameters
    auto parameters = Data::Parameters_Method_GNEB{};

    // Maximum wall time
    std::string str_max_walltime = "0";

    // Chain output filetype
    int output_chain_filetype = static_cast<int>( parameters.output_vf_filetype );

    // Parse
    Log( Log_Level::Debug, Log_Sender::IO, "Parameters GNEB: building" );
    if( !config_file_name.empty() )
    {
        try
        {
            IO::Filter_File_Handle config_file_handle( config_file_name );

            // Output parameters
            config_file_handle.Read_Single( parameters.output_file_tag, "output_file_tag" );
            config_file_handle.Read_Single( parameters.output_folder, "gneb_output_folder" );
            config_file_handle.Read_Single( parameters.output_any, "gneb_output_any" );
            config_file_handle.Read_Single( parameters.output_initial, "gneb_output_initial" );
            config_file_handle.Read_Single( parameters.output_final, "gneb_output_final" );
            config_file_handle.Read_Single( parameters.output_energies_step, "gneb_output_energies_step" );
            config_file_handle.Read_Single(
                parameters.output_energies_add_readability_lines, "gneb_output_energies_add_readability_lines" );
            config_file_handle.Read_Single(
                parameters.output_energies_interpolated, "gneb_output_energies_interpolated" );
            config_file_handle.Read_Single(
                parameters.output_energies_divide_by_nspins, "gneb_output_energies_divide_by_nspins" );
            config_file_handle.Read_Single( parameters.output_chain_step, "gneb_output_chain_step" );
            config_file_handle.Read_Single( output_chain_filetype, "gneb_output_chain_filetype" );
            // Method parameters
            config_file_handle.Read_Single( str_max_walltime, "gneb_max_walltime" );
            config_file_handle.Read_Single( parameters.spring_constant, "gneb_spring_constant" );
            config_file_handle.Read_Single( parameters.force_convergence, "gneb_force_convergence" );
            config_file_handle.Read_Single( parameters.n_iterations, "gneb_n_iterations" );
            config_file_handle.Read_Single( parameters.n_iterations_log, "gneb_n_iterations_log" );
            config_file_handle.Read_Single( parameters.n_iterations_amortize, "gneb_n_iterations_amortize" );
            config_file_handle.Read_Single( parameters.n_E_interpolations, "gneb_n_energy_interpolations" );
            config_file_handle.Read_Single( parameters.moving_endpoints, "gneb_moving_endpoints" );
            config_file_handle.Read_Single( parameters.equilibrium_delta_Rx_left, "gneb_equilibrium_delta_Rx_left" );
            config_file_handle.Read_Single( parameters.equilibrium_delta_Rx_right, "gneb_equilibrium_delta_Rx_right" );
            config_file_handle.Read_Single( parameters.translating_endpoints, "gneb_translating_endpoints" );
        }
        catch( ... )
        {
            spirit_handle_exception_core(
                fmt::format( "Unable to parse GNEB parameters from config file \"{}\"", config_file_name ) );
        }
    }
    auto result = Parameters_Method_GNEB_to_TOML( parameters );
    Apply_Defaults( defaults, result );
    return result;
};

auto Parameters_Method_MMF( const std::string & config_file_name, const IO::Defaults & defaults ) -> toml::table
{
    // Default parameters
    auto parameters = Data::Parameters_Method_MMF{};

    // Maximum wall time
    std::string str_max_walltime = "0";

    // Configuration output filetype
    int output_configuration_filetype = static_cast<int>( parameters.output_vf_filetype );

    // Parse
    Log( Log_Level::Debug, Log_Sender::IO, "Parameters MMF: building" );
    if( !config_file_name.empty() )
    {
        try
        {
            IO::Filter_File_Handle config_file_handle( config_file_name );

            // Output parameters
            config_file_handle.Read_Single( parameters.output_file_tag, "output_file_tag" );
            config_file_handle.Read_Single( parameters.output_folder, "mmf_output_folder" );
            config_file_handle.Read_Single( parameters.output_any, "mmf_output_any" );
            config_file_handle.Read_Single( parameters.output_initial, "mmf_output_initial" );
            config_file_handle.Read_Single( parameters.output_final, "mmf_output_final" );
            config_file_handle.Read_Single( parameters.output_energy_step, "mmf_output_energy_step" );
            config_file_handle.Read_Single( parameters.output_energy_archive, "mmf_output_energy_archive" );
            config_file_handle.Read_Single(
                parameters.output_energy_divide_by_nspins, "mmf_output_energy_divide_by_nspins" );
            config_file_handle.Read_Single(
                parameters.output_energy_add_readability_lines, "mmf_output_energy_add_readability_lines" );
            config_file_handle.Read_Single( parameters.output_configuration_step, "mmf_output_configuration_step" );
            config_file_handle.Read_Single(
                parameters.output_configuration_archive, "mmf_output_configuration_archive" );
            config_file_handle.Read_Single( output_configuration_filetype, "mmf_output_configuration_filetype" );
            // Method parameters
            config_file_handle.Read_Single( str_max_walltime, "mmf_max_walltime" );
            config_file_handle.Read_Single( parameters.force_convergence, "mmf_force_convergence" );
            config_file_handle.Read_Single( parameters.n_iterations, "mmf_n_iterations" );
            config_file_handle.Read_Single( parameters.n_iterations_log, "mmf_n_iterations_log" );
            config_file_handle.Read_Single( parameters.n_iterations_amortize, "mmf_n_iterations_amortize" );
            config_file_handle.Read_Single( parameters.n_modes, "mmf_n_modes" );
            config_file_handle.Read_Single( parameters.n_mode_follow, "mmf_n_mode_follow" );
        }
        catch( ... )
        {
            spirit_handle_exception_core(
                fmt::format( "Unable to parse MMF parameters from config file \"{}\"", config_file_name ) );
        }
    }

    auto result = Parameters_Method_MMF_to_TOML( parameters );
    Apply_Defaults( defaults, result );
    return result;
};

auto Parameters_Method_LLG( const std::string & config_file_name, const IO::Defaults & defaults ) -> toml::table
{
    Data::Parameters_Method_LLG parameters{};

    std::string str_max_walltime      = "0";
    int output_configuration_filetype = static_cast<int>( parameters.output_vf_filetype );

    if( !config_file_name.empty() )
    {
        try
        {
            IO::Filter_File_Handle config_file_handle( config_file_name );

            // Output parameters
            config_file_handle.Read_Single( parameters.output_file_tag, "output_file_tag" );
            config_file_handle.Read_Single( parameters.output_folder, "llg_output_folder" );
            config_file_handle.Read_Single( parameters.output_any, "llg_output_any" );
            config_file_handle.Read_Single( parameters.output_initial, "llg_output_initial" );
            config_file_handle.Read_Single( parameters.output_final, "llg_output_final" );
            config_file_handle.Read_Single( parameters.output_energy_spin_resolved, "llg_output_energy_spin_resolved" );
            config_file_handle.Read_Single( parameters.output_energy_step, "llg_output_energy_step" );
            config_file_handle.Read_Single( parameters.output_energy_archive, "llg_output_energy_archive" );
            config_file_handle.Read_Single(
                parameters.output_energy_divide_by_nspins, "llg_output_energy_divide_by_nspins" );
            config_file_handle.Read_Single(
                parameters.output_energy_add_readability_lines, "llg_output_energy_add_readability_lines" );
            config_file_handle.Read_Single( parameters.output_configuration_step, "llg_output_configuration_step" );
            config_file_handle.Read_Single(
                parameters.output_configuration_archive, "llg_output_configuration_archive" );
            config_file_handle.Read_Single( output_configuration_filetype, "llg_output_configuration_filetype" );
            // Method parameters
            config_file_handle.Read_Single( str_max_walltime, "llg_max_walltime" );
            config_file_handle.Read_Single( parameters.rng_seed, "llg_seed" );
            config_file_handle.Read_Single( parameters.n_iterations, "llg_n_iterations" );
            config_file_handle.Read_Single( parameters.n_iterations_log, "llg_n_iterations_log" );
            config_file_handle.Read_Single( parameters.n_iterations_amortize, "llg_n_iterations_amortize" );
            config_file_handle.Read_Single( parameters.dt, "llg_dt" );
            config_file_handle.Read_Single( parameters.temperature, "llg_temperature" );
            config_file_handle.Read_Vector3(
                parameters.temperature_gradient_direction, "llg_temperature_gradient_direction" );
            config_file_handle.Read_Single(
                parameters.temperature_gradient_inclination, "llg_temperature_gradient_inclination" );
            config_file_handle.Read_Single( parameters.damping, "llg_damping" );
            config_file_handle.Read_Single( parameters.beta, "llg_beta" );
            // config_file_handle.Read_Single(parameters.renorm_sd, "llg_renorm");
            config_file_handle.Read_Single( parameters.stt_use_gradient, "llg_stt_use_gradient" );
            config_file_handle.Read_Single( parameters.stt_magnitude, "llg_stt_magnitude" );
            config_file_handle.Read_Vector3( parameters.stt_polarisation_normal, "llg_stt_polarisation_normal" );
            config_file_handle.Read_Single( parameters.force_convergence, "llg_force_convergence" );
        }
        catch( ... )
        {
            spirit_handle_exception_core(
                fmt::format( "Unable to parse LLG parameters from config file \"{}\"", config_file_name ) );
        }
    }
    auto result = Parameters_Method_LLG_to_TOML( parameters );
    Apply_Defaults( defaults, result );
    return result;
};

auto Parameters_Method_MC( const std::string & config_file_name, const IO::Defaults & defaults ) -> toml::table
{
    // Default parameters
    auto parameters = Data::Parameters_Method_MC{};

    // PRNG Seed
    std::random_device random;
    parameters.rng_seed = random();

    // Maximum wall time
    std::string str_max_walltime = "0";

    // Metropolis Step
    std::string metropolis_step = "cone";

    // Configuration output filetype
    int output_configuration_filetype = static_cast<int>( parameters.output_vf_filetype );

    // Parse
    if( !config_file_name.empty() )
    {
        try
        {
            IO::Filter_File_Handle config_file_handle( config_file_name );

            // Output parameters
            config_file_handle.Read_Single( parameters.output_file_tag, "output_file_tag" );
            config_file_handle.Read_Single( parameters.output_folder, "mc_output_folder" );
            config_file_handle.Read_Single( parameters.output_any, "mc_output_any" );
            config_file_handle.Read_Single( parameters.output_initial, "mc_output_initial" );
            config_file_handle.Read_Single( parameters.output_final, "mc_output_final" );
            config_file_handle.Read_Single( parameters.output_energy_spin_resolved, "mc_output_energy_spin_resolved" );
            config_file_handle.Read_Single( parameters.output_energy_step, "mc_output_energy_step" );
            config_file_handle.Read_Single( parameters.output_energy_archive, "mc_output_energy_archive" );
            config_file_handle.Read_Single(
                parameters.output_energy_divide_by_nspins, "mc_output_energy_divide_by_nspins" );
            config_file_handle.Read_Single(
                parameters.output_energy_add_readability_lines, "mc_output_energy_add_readability_lines" );
            config_file_handle.Read_Single( parameters.output_configuration_step, "mc_output_configuration_step" );
            config_file_handle.Read_Single(
                parameters.output_configuration_archive, "mc_output_configuration_archive" );
            config_file_handle.Read_Single( output_configuration_filetype, "mc_output_configuration_filetype" );
            // Method parameters
            config_file_handle.Read_Single( str_max_walltime, "mc_max_walltime" );
            config_file_handle.Read_Single( parameters.rng_seed, "mc_seed" );
            config_file_handle.Read_Single( parameters.n_iterations, "mc_n_iterations" );
            config_file_handle.Read_Single( parameters.n_iterations_log, "mc_n_iterations_log" );
            config_file_handle.Read_Single( parameters.n_iterations_amortize, "mc_n_iterations_amortize" );
            config_file_handle.Read_Single( parameters.temperature, "mc_temperature" );
            config_file_handle.Read_Single( metropolis_step, "mc_metropolis_step" );
            config_file_handle.Read_Single( parameters.metropolis_cone_adaptive, "mc_metropolis_use_adaptive_cone" );
            config_file_handle.Read_Single( parameters.acceptance_ratio_target, "mc_acceptance_ratio" );
            config_file_handle.Read_Single( parameters.metropolis_cone_angle, "mc_metropolis_cone_angle" );
            config_file_handle.Read_Single( parameters.metropolis_random_sample, "mc_metropolis_random_sample" );
        }
        catch( ... )
        {
            spirit_handle_exception_core(
                fmt::format( "Unable to parse MC parameters from config file \"{}\"", config_file_name ) );
        }
    }
    else
        Log( Log_Level::Parameter, Log_Sender::IO, "Parameters MC: Using default configuration!" );
    auto result = Parameters_Method_MC_to_TOML( parameters );
    Apply_Defaults( defaults, result );
    return result;
};

auto Boundary_Conditions( const std::string & config_file_name ) -> toml::table
try
{
    // Boundary conditions (a, b, c)
    std::vector<int> boundary_conditions_i = { 0, 0, 0 };
    intfield boundary_conditions           = { false, false, false };

    if( !config_file_name.empty() )
    {
        IO::Filter_File_Handle config_file_handle( config_file_name );

        // Boundary conditions
        config_file_handle.Read_3Vector( boundary_conditions_i, "boundary_conditions" );
        boundary_conditions[0] = static_cast<int>( boundary_conditions_i[0] != 0 );
        boundary_conditions[1] = static_cast<int>( boundary_conditions_i[1] != 0 );
        boundary_conditions[2] = static_cast<int>( boundary_conditions_i[2] != 0 );
    }

    return toml::table{ { "boundary_conditions", toml_array_from_container( boundary_conditions ) } };
}
catch( ... )
{
    spirit_rethrow( fmt::format( "Unable to parse boundary conditions from config file \"{}\"", config_file_name ) );
    return toml::table{};
} // End boundary_conditions from Config

auto Bravais_Vectors( const std::string & config_file_name ) -> toml::table
{
    toml::table result{};
    try
    {
        std::vector<Vector3> bravais_vectors{ { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
        IO::Filter_File_Handle config_file_handle( config_file_name );

        // Bravais lattice type or manually specified vectors/matrix
        if( config_file_handle.Find( "bravais_lattice" ) )
        {
            std::string bravais_lattice = "";
            config_file_handle >> bravais_lattice;
            result.insert( "bravais_lattice", bravais_lattice );
        }
        if( config_file_handle.Find( "bravais_vectors" ) )
        {
            config_file_handle.GetLine();
            config_file_handle >> bravais_vectors[0][0] >> bravais_vectors[0][1] >> bravais_vectors[0][2];
            config_file_handle.GetLine();
            config_file_handle >> bravais_vectors[1][0] >> bravais_vectors[1][1] >> bravais_vectors[1][2];
            config_file_handle.GetLine();
            config_file_handle >> bravais_vectors[2][0] >> bravais_vectors[2][1] >> bravais_vectors[2][2];
            result.insert( "bravais_vectors", toml_array_from_container( bravais_vectors ) );
        }
        if( config_file_handle.Find( "bravais_matrix" ) )
        {
            config_file_handle.GetLine();
            config_file_handle >> bravais_vectors[0][0] >> bravais_vectors[1][0] >> bravais_vectors[2][0];
            config_file_handle.GetLine();
            config_file_handle >> bravais_vectors[0][1] >> bravais_vectors[1][1] >> bravais_vectors[2][1];
            config_file_handle.GetLine();
            config_file_handle >> bravais_vectors[0][2] >> bravais_vectors[1][2] >> bravais_vectors[2][2];
            result.insert( "bravais_matrix", toml_array_from_container( bravais_vectors ) );
        }
    }
    catch( ... )
    {
        spirit_rethrow( fmt::format( "Unable to parse bravais vectors from config file \"{}\"", config_file_name ) );
    } // End Basis_from_Config

    return result;
}

auto Pinning( const std::string & config_file_name, std::size_t n_cell_atoms ) -> toml::table
{
#ifndef SPIRIT_ENABLE_PINNING
    Log( Log_Level::Parameter, Log_Sender::IO, "Pinning is disabled" );
    if( !config_file_name.empty() )
    {
        try
        {
            IO::Filter_File_Handle config_file_handle( config_file_name );
            if( config_file_handle.Find( "pinning_cell" ) )
                Log( Log_Level::Warning, Log_Sender::IO,
                     "You specified a pinning cell even though pinning is disabled!" );
        }
        catch( ... )
        {
            spirit_handle_exception_core( fmt::format(
                "Failed to read pinning parameters from file \"{}\". Leaving values at default.", config_file_name ) );
        }
    }

    return Data::Pinning{ 0, 0, 0, 0, 0, 0, vectorfield( 0 ), field<Site>( 0 ), vectorfield( 0 ) };
#else
    vectorfield pinned_cell( n_cell_atoms, Vector3{ 0, 0, 1 } );
    //-------------- Insert default values here -----------------------------
    int na = 0, na_left = 0, na_right = 0;
    int nb = 0, nb_left = 0, nb_right = 0;
    int nc = 0, nc_left = 0, nc_right = 0;
    // Additional pinned sites
    field<Site> pinned_sites( 0 );
    vectorfield pinned_spins( 0 );
    int n_pinned = 0;

    toml::table tbl;

    //------------------------------- Parser --------------------------------
    Log( Log_Level::Debug, Log_Sender::IO, "going to read pinning" );
    if( !config_file_name.empty() )
    {
        try
        {
            Filter_File_Handle config_file_handle( config_file_name );

            {
                toml::array pinning_boundary{};
                // N_a
                config_file_handle.Read_Single( na_left, "pin_na_left", false );
                config_file_handle.Read_Single( na_right, "pin_na_right", false );
                config_file_handle.Read_Single( na, "pin_na ", false );

                if( na_left == 0 || na_right == 0 )
                    pinning_boundary.push_back( na );
                else
                    pinning_boundary.push_back( toml::array{ na_left, na_right } );

                // N_b
                config_file_handle.Read_Single( nb_left, "pin_nb_left", false );
                config_file_handle.Read_Single( nb_right, "pin_nb_right", false );
                config_file_handle.Read_Single( nb, "pin_nb ", false );
                if( nb_left == 0 || nb_right == 0 )
                    pinning_boundary.push_back( nb );
                else
                    pinning_boundary.push_back( toml::array{ nb_left, nb_right } );

                // N_c
                config_file_handle.Read_Single( nc_left, "pin_nc_left", false );
                config_file_handle.Read_Single( nc_right, "pin_nc_right", false );
                config_file_handle.Read_Single( nc, "pin_nc ", false );
                if( nc_left == 0 || nc_right == 0 )
                    pinning_boundary.push_back( nc );
                else
                    pinning_boundary.push_back( toml::array{ nc_left, nc_right } );

                const auto specified = []( const auto & node )
                {
                    auto value = node.as_integer();
                    return !value || *value != 0;
                };
                if( std::any_of( pinning_boundary.begin(), pinning_boundary.end(), specified ) )
                    tbl.insert( "boundary", pinning_boundary );
            };

            // How should the cells be pinned
            if( config_file_handle.Find( "pinning_cell" ) )
            {
                for( std::size_t i = 0; i < n_cell_atoms; ++i )
                {
                    config_file_handle.GetLine();
                    config_file_handle >> pinned_cell[i][0] >> pinned_cell[i][1] >> pinned_cell[i][2];
                }
                tbl.insert( "pinning_cell", toml_array_from_container( pinned_cell ) );
            }

            // Additional pinned sites
            std::string pinned_file = "";
            if( config_file_handle.Find( "n_pinned" ) )
            {
                config_file_handle.Read_Single( n_pinned, "n_pinned" );
                std::stringstream oss{ "\n" };
                for( int i = 0; i < n_pinned; ++i )
                {
                    if( !config_file_handle.GetLine() )
                        break;
                    oss << config_file_handle.CurrentLine() << '\n';
                }
                tbl.insert( "pinned", oss.str() );
            }
            else if( config_file_handle.Find( "pinned_from_file" ) )
            {
                config_file_handle >> pinned_file;
                tbl.insert( "pinned", fmt::format( "{}{}", Filter_File_Handle::file_prefix, pinned_file ) );
            }
        }
        catch( ... )
        {
            spirit_handle_exception_core( fmt::format(
                "Failed to read Pinning from file \"{}\". Leaving values at default.", config_file_name ) );
        }
    }

    // Return Pinning
    Log( Log_Level::Debug, Log_Sender::IO, "pinning has been read" );
    return tbl;
#endif // SPIRIT_ENABLE_PINNING
}

auto Basis_Cell_Composition( const std::string & config_file_name, const std::size_t n_cell_atoms ) -> toml::table
{
    toml::table tbl;
    auto cell_composition = Data::Basis_Cell_Composition::make_default( n_cell_atoms, /*disordered=*/false );
    try
    {
        IO::Filter_File_Handle config_file_handle( config_file_name );

        // Spin moment
        if( !config_file_handle.Find( "atom_types" ) )
        {
            if( config_file_handle.Find( "mu_s" ) )
            {
                for( std::size_t iatom = 0; iatom < n_cell_atoms; ++iatom )
                {
                    if( !( config_file_handle >> cell_composition.mu_s[iatom] ) )
                    {
                        Log( Log_Level::Warning, Log_Sender::IO,
                             fmt::format(
                                 "Not enough values specified after 'mu_s'. Expected {}. Using "
                                 "mu_s[{}]=mu_s[0]={}",
                                 n_cell_atoms, iatom, cell_composition.mu_s[0] ) );
                        cell_composition.mu_s[iatom] = cell_composition.mu_s[0];
                    }
                }

                tbl.insert( "mu_s", toml_array_from_container( cell_composition.mu_s ) );
            }

            if( config_file_handle.Find( "spin_qn" ) )
            {
                for( std::size_t iatom = 0; iatom < n_cell_atoms; ++iatom )
                {
                    if( !( config_file_handle >> cell_composition.spin_qn[iatom] ) )
                    {
                        Log( Log_Level::Warning, Log_Sender::IO,
                             fmt::format(
                                 "Not enough values specified after 'spin_qn'. Expected {}. Using "
                                 "spin_qn[{}]=spin_qn[0]={}",
                                 n_cell_atoms, iatom, cell_composition.spin_qn[0] ) );
                        cell_composition.spin_qn[iatom] = cell_composition.spin_qn[0];
                    }
                }
                tbl.insert( "spin_qn", toml_array_from_container( cell_composition.spin_qn ) );
            }
        }
    }
    catch( ... )
    {
        spirit_handle_exception_core( fmt::format( "Unable to read mu_s from config file \"{}\"", config_file_name ) );
    }

    // Defects
#ifdef SPIRIT_ENABLE_DEFECTS
    try
    {
        IO::Filter_File_Handle config_file_handle( config_file_name );

        int n_defects = 0;

        std::string defects_file = "";
        if( config_file_handle.Find( "n_defects" ) )
        {
            config_file_handle >> n_defects;
            std::stringstream oss{ "\n" };
            for( int i = 0; i < n_defects; ++i )
            {
                if( !config_file_handle.GetLine() )
                    break;
                oss << config_file_handle.CurrentLine() << '\n';
            }
            tbl.insert( "defects", oss.str() );
        }
        else if( config_file_handle.Find( "defects_from_file" ) )
        {
            config_file_handle >> defects_file;
            tbl.insert( "defects", fmt::format( "{}{}", Filter_File_Handle::file_prefix, defects_file ) );
        }

        // Disorder
        if( config_file_handle.Find( "atom_types" ) )
        {
            config_file_handle >> n_atom_types;
            std::stringstream oss{ "\n" };
            for( int i = 0; i < n_atom_types; ++i )
            {
                if( !config_file_handle.GetLine() )
                    break;
                oss << config_file_handle.CurrentLine() << '\n';
            }
            tbl.insert( "atom_types", oss.str() );
        }
    }
    catch( ... )
    {
        spirit_handle_exception_core( fmt::format(
            "Failed to read defect parameters from file \"{}\". Leaving values at default.", config_file_name ) );
    }
#else
    Log( Log_Level::Parameter, Log_Sender::IO, "Disorder is disabled" );
#endif
    return tbl;
}

auto Geometry( const std::string & config_file_name ) -> toml::table
{
    toml::table tbl{};
    //-------------- Insert default values here -----------------------------
    // Atoms in the basis
    std::size_t n_cell_atoms = 1;
    // Lattice Constant [Angstrom]
    scalar lattice_constant = 1;
    // Number of translations nT for each basis direction
    intfield n_cells = { 100, 100, 1 };

    try
    {
        //------------------------------- Parser --------------------------------
        Log( Log_Level::Debug, Log_Sender::IO, "Geometry: building" );
        if( !config_file_name.empty() )
        {
            try
            {
                IO::Filter_File_Handle config_file_handle( config_file_name );

                // Lattice constant
                config_file_handle.Read_Single( lattice_constant, "lattice_constant" );
                tbl.insert( "lattice_constant", lattice_constant );

                // Get the bravais lattice type and vectors
                {
                    auto bravais = Bravais_Vectors( config_file_name );
                    tbl.insert( bravais.begin(), bravais.end() );
                }
                // Read number of basis cells
                config_file_handle.Read_3Vector( n_cells, "n_basis_cells" );
                tbl.insert( "n_basis_cells", toml_array_from_container( n_cells ) );

                // Basis
                if( config_file_handle.Find( "basis_file" ) )
                {
                    std::string basis_file = "";
                    config_file_handle >> basis_file;
                    Filter_File_Handle( basis_file ).Read_Single( n_cell_atoms, "n_basis" );
                    tbl.insert( "basis", fmt::format( "{}{}", Filter_File_Handle::file_prefix, basis_file ) );
                }
                else if( config_file_handle.Find( "basis" ) )
                {
                    config_file_handle.GetLine();
                    config_file_handle >> n_cell_atoms;
                    std::ostringstream oss;
                    for( unsigned int i = 0; i < n_cell_atoms; ++i )
                    {
                        if( !config_file_handle.GetLine() )
                            break;
                        oss << config_file_handle.CurrentLine() << '\n';
                    }
                    tbl.insert( "basis", fmt::format( "\n{}\n{}", n_cell_atoms, oss.str() ) );
                }
            }
            catch( ... )
            {
                spirit_handle_exception_core( fmt::format(
                    "Failed to read Geometry parameters from file \"{}\". Leaving values at default.",
                    config_file_name ) );
            }

            {
                const auto composition_table = Basis_Cell_Composition( config_file_name, n_cell_atoms );
                tbl.insert( composition_table.begin(), composition_table.end() );
            }
        }
        return tbl;
    }
    catch( ... )
    {
        spirit_rethrow( fmt::format( "Unable to parse geometry from config file \"{}\"", config_file_name ) );
        return toml::table{};
    }
} // End Geometry from Config

namespace Interaction
{

auto Anisotropy( const std::string & config_file_name ) -> toml::table
{
    toml::table tbl{};

    scalar K = 0, K4 = 0;
    Vector3 K_normal = { 0, 0, 0 };

    try
    {
        IO::Filter_File_Handle config_file_handle( config_file_name );
        if( config_file_handle.Find( "n_anisotropy" ) )
        {
            int n_anisotropy = 0;
            config_file_handle >> n_anisotropy;
            tbl.insert(
                "anisotropy",
                [n_anisotropy, &config_file_handle]() -> std::string
                {
                    if( n_anisotropy <= 0 )
                        return "";

                    std::stringstream oss;
                    oss << '\n';
                    for( int i = 0; i < n_anisotropy + 1; ++i )
                    {
                        if( !config_file_handle.GetLine() )
                            break;
                        oss << config_file_handle.CurrentLine() << '\n';
                    }
                    return oss.str();
                }() );
        }
        else if( config_file_handle.Find( "anisotropy_file" ) )
        {
            std::string anisotropy_file = "";
            config_file_handle >> anisotropy_file;
            tbl.insert( "anisotropy", fmt::format( "{}{}", Filter_File_Handle::file_prefix, anisotropy_file ) );
        }
        else
        {
            // Read parameters from config
            config_file_handle.Read_Single( K, "anisotropy_magnitude" );
            tbl.insert( "anisotropy_magnitude", K );

            config_file_handle.Read_Vector3( K_normal, "anisotropy_normal" );
            tbl.insert( "anisotropy_normal", toml_array_from_container( K_normal ) );

            config_file_handle.Read_Single( K4, "cubic_anisotropy_magnitude" );
            tbl.insert( "cubic_anisotropy_magnitude", K4 );
        }
    }
    catch( ... )
    {
        spirit_handle_exception_core(
            fmt::format( "Unable to read anisotropy from config file \"{}\"", config_file_name ) );
    }

    return tbl;
}

auto Biaxial_Anisotropy( const std::string & config_file_name ) -> toml::table
{
    toml::table tbl;
    try
    {
        IO::Filter_File_Handle config_file_handle( config_file_name );

        if( config_file_handle.Find( "n_biaxial_anisotropy_axes" ) )
        {
            int n_biaxial_anisotropy_axes = 0;
            config_file_handle >> n_biaxial_anisotropy_axes;
            tbl.insert(
                "biaxial_anisotropy_axes",
                [n_biaxial_anisotropy_axes, &config_file_handle]() -> std::string
                {
                    if( n_biaxial_anisotropy_axes == 0 )
                        return "";

                    std::stringstream oss;
                    oss << '\n';
                    for( int i = 0; i < n_biaxial_anisotropy_axes + 1; ++i )
                    {
                        if( !config_file_handle.GetLine() )
                            break;
                        oss << config_file_handle.CurrentLine() << '\n';
                    }
                    return oss.str();
                }() );
        }
        else if( config_file_handle.Find( "biaxial_anisotropy_axes_file" ) )
        {
            std::string biaxial_anisotropy_axes_file = "";
            config_file_handle >> biaxial_anisotropy_axes_file;
            tbl.insert(
                "biaxial_anisotropy_axes",
                fmt::format( "{}{}", Filter_File_Handle::file_prefix, biaxial_anisotropy_axes_file ) );
        }

        if( config_file_handle.Find( "n_biaxial_anisotropy_terms" ) )
        {
            int n_biaxial_anisotropy_terms = 0;
            config_file_handle >> n_biaxial_anisotropy_terms;
            tbl.insert(
                "biaxial_anisotropy_terms",
                [n_biaxial_anisotropy_terms, &config_file_handle]() -> std::string
                {
                    if( n_biaxial_anisotropy_terms == 0 )
                        return "";

                    std::stringstream oss;
                    oss << '\n';
                    for( int i = 0; i < n_biaxial_anisotropy_terms + 1; ++i )
                    {
                        if( !config_file_handle.GetLine() )
                            break;
                        oss << config_file_handle.CurrentLine() << '\n';
                    }
                    return oss.str();
                }() );
        }
        else if( config_file_handle.Find( "biaxial_anisotropy_terms_file" ) )
        {
            std::string biaxial_anisotropy_terms_file = "";
            config_file_handle >> biaxial_anisotropy_terms_file;
            tbl.insert(
                "biaxial_anisotropy_terms",
                fmt::format( "{}{}", Filter_File_Handle::file_prefix, biaxial_anisotropy_terms_file ) );
        }
    }
    catch( ... )
    {
        spirit_handle_exception_core(
            fmt::format( "Could not read biaxial anisotropy from config \"{}\"", config_file_name ) );
    }

    return tbl;
}

auto DDI( const std::string & config_file_name ) -> toml::table
{
    std::string ddi_method_str{};
    intfield ddi_n_periodic_images = { 4, 4, 4 };
    bool ddi_pb_zero_padding       = false;
    scalar ddi_radius              = 0.0;

    try
    {
        IO::Filter_File_Handle config_file_handle( config_file_name );

        config_file_handle.Read_String( ddi_method_str, "ddi_method" );
        config_file_handle.Read_3Vector( ddi_n_periodic_images, "ddi_n_periodic_images" );
        config_file_handle.Read_Single( ddi_pb_zero_padding, "ddi_pb_zero_padding" );
        config_file_handle.Read_Single( ddi_radius, "ddi_radius" );
    }
    catch( ... )
    {
        spirit_handle_exception_core(
            fmt::format( "Unable to read DDI radius from config file \"{}\"", config_file_name ) );
    };

    if( ddi_method_str.empty() || ddi_radius == 0 )
        return toml::table{};
    else
        return toml::table{
            { "ddi_method", ddi_method_str },
            { "ddi_radius", ddi_radius },
            { "ddi_n_periodic_images", toml_array_from_container( ddi_n_periodic_images ) },
            { "ddi_pb_zero_padding", ddi_pb_zero_padding },
        };
}

auto Gaussian( const std::string & config_file_name ) -> toml::table
{
    toml::table tbl;

    try
    {
        IO::Filter_File_Handle config_file_handle( config_file_name );

        // N
        int n_gaussians = 0;
        config_file_handle.Read_Single( n_gaussians, "n_gaussians" );
        if( n_gaussians <= 0 )
            return tbl;

        tbl.insert(
            "gaussians",
            [n_gaussians, &config_file_handle]() -> std::string
            {
                std::ostringstream oss;
                oss << "\n";

                int i = 0;
                if( config_file_handle.Find( "gaussians" ) )
                {
                    for( ; i < n_gaussians; ++i )
                    {
                        if( !config_file_handle.GetLine() )
                            break;
                        oss << config_file_handle.CurrentLine() << '\n';
                    }
                }
                else
                    Log( Log_Level::Error, Log_Sender::IO,
                         "Hamiltonian_Gaussian: Keyword 'gaussians' not found. Using Default: 1.0 1.0 {0, 0, 1}" );

                // pad missing lines with the default value
                for( ; i < n_gaussians; ++i )
                    oss << "1.0  1.0  0 0 1\n";

                return oss.str();
            }() );
    }
    catch( ... )
    {
        spirit_handle_exception_core( fmt::format(
            "Unable to read Hamiltonian_Gaussian parameters from config file  \"{}\"", config_file_name ) );
    }

    return tbl;
}

auto Pair_Interactions_from_Pairs( const std::string & config_file_name ) -> toml::table
{
    toml::table tbl{};

    try
    {
        IO::Filter_File_Handle config_file_handle( config_file_name );

        // Interaction Pairs
        if( config_file_handle.Find( "n_interaction_pairs" ) )
        {
            int n_pairs = 0;
            config_file_handle >> n_pairs;
            tbl.insert(
                "pairs",
                [n_pairs, &config_file_handle]() -> std::string
                {
                    if( n_pairs <= 0 )
                        return "";

                    std::stringstream oss;
                    oss << '\n';
                    for( int i = 0; i < n_pairs + 1; ++i )
                    {
                        if( !config_file_handle.GetLine() )
                            break;
                        oss << config_file_handle.CurrentLine() << '\n';
                    }
                    return oss.str();
                }() );
        }
        else if( config_file_handle.Find( "interaction_pairs_file" ) )
        {
            std::string pairs_file = "";
            config_file_handle >> pairs_file;
            tbl.insert( "interaction_pairs", fmt::format( "{}{}", Filter_File_Handle::file_prefix, pairs_file ) );
        }
    }
    catch( ... )
    {
        spirit_handle_exception_core(
            fmt::format( "Unable to read interaction pairs from config file \"{}\"", config_file_name ) );
    }

    return tbl;
}

auto Pair_Interactions_from_Shells( const std::string & config_file_name ) -> toml::table
{
    toml::table tbl{};

    try
    {
        IO::Filter_File_Handle config_file_handle( config_file_name );

        int n_shells_exchange = 0;
        config_file_handle.Read_Single( n_shells_exchange, "n_shells_exchange" );

        if( n_shells_exchange > 0 )
        {
            auto exchange_magnitudes = scalarfield( n_shells_exchange );
            if( config_file_handle.Find( "jij" ) )
            {
                for( int ishell = 0; ishell < n_shells_exchange; ++ishell )
                    config_file_handle >> exchange_magnitudes[ishell];
            }
            else
                Log( Log_Level::Warning, Log_Sender::IO,
                     fmt::format(
                         "Hamiltonian_Heisenberg: Keyword 'jij' not found. Using Default: {}",
                         exchange_magnitudes[0] ) );

            tbl.insert( "exchange_shells", toml_array_from_container( exchange_magnitudes ) );
        }
    }
    catch( ... )
    {
        spirit_handle_exception_core(
            fmt::format( "Failed to read exchange parameters from config file \"{}\"", config_file_name ) );
    }

    try
    {
        IO::Filter_File_Handle config_file_handle( config_file_name );

        int n_shells_dmi = 0;
        int dm_chirality = 0;
        config_file_handle.Read_Single( n_shells_dmi, "n_shells_dmi" );
        if( n_shells_dmi > 0 )
        {
            auto dmi_magnitudes = scalarfield( n_shells_dmi );
            if( config_file_handle.Find( "dij" ) )
            {
                for( int ishell = 0; ishell < n_shells_dmi; ++ishell )
                    config_file_handle >> dmi_magnitudes[ishell];
            }
            else
                Log( Log_Level::Warning, Log_Sender::IO,
                     fmt::format(
                         "Hamiltonian_Heisenberg: Keyword 'dij' not found. Using Default: {}", dmi_magnitudes[0] ) );
            tbl.insert( "dmi_shells", toml_array_from_container( dmi_magnitudes ) );
        }

        config_file_handle.Read_Single( dm_chirality, "dm_chirality" );
        tbl.insert( "dmi_chirality", dm_chirality );
    }
    catch( ... )
    {
        spirit_handle_exception_core(
            fmt::format( "Failed to read DMI parameters from config file \"{}\"", config_file_name ) );
    }

    return tbl;
}

auto Quadruplets( const std::string & config_file_name ) -> toml::table
{
    toml::table tbl{};

    try
    {
        IO::Filter_File_Handle config_file_handle( config_file_name );

        if( config_file_handle.Find( "n_interaction_quadruplets" ) )
        {
            int n_quadruplets = 0;
            config_file_handle >> n_quadruplets;
            tbl.insert(
                "quadruplets",
                [n_quadruplets, &config_file_handle]() -> std::string
                {
                    if( n_quadruplets <= 0 )
                        return "";

                    std::stringstream oss;
                    oss << '\n';
                    for( int i = 0; i < n_quadruplets + 1; ++i )
                    {
                        if( !config_file_handle.GetLine() )
                            break;
                        oss << config_file_handle.CurrentLine() << '\n';
                    }
                    return oss.str();
                }() );
        }
        else if( config_file_handle.Find( "interaction_quadruplets_file" ) )
        {
            std::string quadruplets_file = "";
            config_file_handle >> quadruplets_file;
            tbl.insert( "quadruplets", fmt::format( "{}{}", Filter_File_Handle::file_prefix, quadruplets_file ) );
        }
    }
    catch( ... )
    {
        spirit_handle_exception_core(
            fmt::format( "Unable to read interaction quadruplets from config file \"{}\"", config_file_name ) );
    }

    return tbl;
}

auto Zeeman( const std::string & config_file_name ) -> toml::table
{
    scalar magnitude = 0.0;
    Vector3 normal   = { 0.0, 0.0, 1.0 };

    try
    {
        IO::Filter_File_Handle config_file_handle( config_file_name );

        // Read parameters from config if available
        config_file_handle.Read_Single( magnitude, "external_field_magnitude" );
        config_file_handle.Read_Vector3( normal, "external_field_normal" );
    }
    catch( ... )
    {
        spirit_handle_exception_core(
            fmt::format( "Unable to read external field from config file \"{}\"", config_file_name ) );
    }

    return toml::table{
        { "external_field_magnitude", magnitude },
        { "external_field_normal", toml_array_from_container( normal ) },
    };
}

} // namespace Interaction

std::string Hamiltonian_Type( const std::string & config_file_name, const std::string_view default_type )
{
    std::string hamiltonian_type{ default_type };

    // Hamiltonian type
    if( !config_file_name.empty() )
    {
        try
        {
            Log( Log_Level::Debug, Log_Sender::IO, "Hamiltonian: deciding type" );
            IO::Filter_File_Handle config_file_handle( config_file_name );

            // What hamiltonian do we use?
            config_file_handle.Read_Single( hamiltonian_type, "hamiltonian" );
        }
        catch( ... )
        {
            spirit_handle_exception_core( fmt::format(
                "Unable to read Hamiltonian type from config file  \"{}\". Using default.", config_file_name ) );
            hamiltonian_type = default_type;
        }
    }
    else
        Log( Log_Level::Parameter, Log_Sender::IO,
             fmt::format( "Hamiltonian: Using default Hamiltonian: {}", hamiltonian_type ) );

    return hamiltonian_type;
}

auto Hamiltonian( const std::string & config_file_name ) -> toml::table
{
    const auto hamiltonian_type = Hamiltonian_Type( config_file_name, "heisenberg_neighbours" );
    const auto extend           = []( toml::table & tbl, toml::table && ext ) { tbl.insert( ext.begin(), ext.end() ); };

    toml::table tbl;
    extend( tbl, Boundary_Conditions( config_file_name ) );
    if( hamiltonian_type == "gaussian" )
    {
        extend( tbl, Interaction::Gaussian( config_file_name ) );
    }
    else if( hamiltonian_type == "heisenberg_neighbours" || hamiltonian_type == "heisenberg_pairs" )
    {
        extend( tbl, Interaction::Zeeman( config_file_name ) );
        extend( tbl, Interaction::Anisotropy( config_file_name ) );
        extend( tbl, Interaction::Biaxial_Anisotropy( config_file_name ) );
        if( hamiltonian_type == "heisenberg_pairs" )
            extend( tbl, Interaction::Pair_Interactions_from_Pairs( config_file_name ) );
        else
            extend( tbl, Interaction::Pair_Interactions_from_Shells( config_file_name ) );
        extend( tbl, Interaction::Quadruplets( config_file_name ) );
        extend( tbl, Interaction::DDI( config_file_name ) );
    }
    else
        spirit_throw(
            Utility::Exception_Classifier::Input_parse_failed, Log_Level::Error,
            fmt::format( "Hamiltonian: Invalid type \"{}\"", hamiltonian_type ) );

    return tbl;
}

} // namespace detail

auto Config( const std::string & config_file_name ) -> toml::table
{
    auto defaults = detail::Defaults( config_file_name );
    auto result   = toml::table{
          { "version", "0.1" },
          { "logging", detail::Logging( config_file_name, defaults ) },
          { "geometry", detail::Geometry( config_file_name ) },
          { "hamiltonian", detail::Hamiltonian( config_file_name ) },
          { "method",
            toml::table{
                { "llg", detail::Parameters_Method_LLG( config_file_name, defaults ) },
                { "mc", detail::Parameters_Method_MC( config_file_name, defaults ) },
                { "mmf", detail::Parameters_Method_MMF( config_file_name, defaults ) },
                { "ema", detail::Parameters_Method_EMA( config_file_name, defaults ) },
                { "gneb", detail::Parameters_Method_LLG( config_file_name, defaults ) },
          } },
    };

    if( defaults.output.file_tag || defaults.output.directory )
    {
        toml::table output{};
        if( defaults.output.file_tag )
            output.insert( "file_tag", *defaults.output.file_tag );

        if( defaults.output.directory )
            output.insert( "folder", *defaults.output.directory );

        result.insert( "defaults", toml::table{ { "output", output } } );
    }
    return result;
}

} // namespace convert

} // namespace IO
