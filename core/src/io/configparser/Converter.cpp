#include <io/Configparser.hpp>
#include <io/Filter_File_Handle.hpp>
#include <io/configparser/Converter.hpp>
#include <utility/Logging.hpp>

#include <toml++/toml.hpp>

#include <string>

namespace IO
{

namespace convert
{

namespace
{

template<typename Container>
auto toml_array_from_container( const Container & iterable ) -> toml::array
{
    toml::array result{};
    result.insert( result.begin(), iterable.begin(), iterable.end() );
    return result;
}

} // namespace

using Utility::Log_Level;
using Utility::Log_Sender;

auto Logging( const std::string & config_file_name ) -> toml::table
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

    return toml::table{
        { "output_file_tag", file_tag },
        { "output_folder", output_folder },
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
}

auto Parameters_Method_EMA( const std::string & config_file_name ) -> toml::table
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

    return toml::table{
        // Output parameters
        { "output_file_tag", parameters.output_file_tag },
        { "output_folder", parameters.output_folder },
        { "output_any", parameters.output_any },
        { "output_initial", parameters.output_initial },
        { "output_final", parameters.output_final },
        { "output_energy_divide_by_nspins", parameters.output_energy_divide_by_nspins },
        { "output_energy_spin_resolved", parameters.output_energy_spin_resolved },
        { "output_energy_step", parameters.output_energy_step },
        { "output_energy_archive", parameters.output_energy_archive },
        { "output_configuration_step", parameters.output_configuration_step },
        { "output_configuration_archive", parameters.output_configuration_archive },
        // Method parameters
        { "max_walltime", str_max_walltime },
        { "n_iterations", parameters.n_iterations },
        { "n_iterations_log", parameters.n_iterations_log },
        { "n_modes", parameters.n_modes },
        { "n_mode_follow", parameters.n_mode_follow },
        { "frequency", parameters.frequency },
        { "amplitude", parameters.amplitude },
        { "sparse", parameters.sparse },

    };
}; // namespace IO

auto Parameters_Method_GNEB( const std::string & config_file_name ) -> toml::table
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

    return toml::table{
        { "output_file_tag", parameters.output_file_tag },
        { "output_folder", parameters.output_folder },
        { "output_any", parameters.output_any },
        { "output_initial", parameters.output_initial },
        { "output_final", parameters.output_final },
        { "output_energies_step", parameters.output_energies_step },
        { "output_energies_add_readability_lines", parameters.output_energies_add_readability_lines },
        { "output_energies_interpolated", parameters.output_energies_interpolated },
        { "output_energies_divide_by_nspins", parameters.output_energies_divide_by_nspins },
        { "output_chain_step", parameters.output_chain_step },
        { "output_chain_filetype", output_chain_filetype },
        // Method parameters
        { "max_walltime", str_max_walltime },
        { "spring_constant", parameters.spring_constant },
        { "force_convergence", parameters.force_convergence },
        { "n_iterations", parameters.n_iterations },
        { "n_iterations_log", parameters.n_iterations_log },
        { "n_iterations_amortize", parameters.n_iterations_amortize },
        { "n_energy_interpolations", parameters.n_E_interpolations },
        { "moving_endpoints", parameters.moving_endpoints },
        { "equilibrium_delta_Rx_left", parameters.equilibrium_delta_Rx_left },
        { "equilibrium_delta_Rx_right", parameters.equilibrium_delta_Rx_right },
        { "translating_endpoints", parameters.translating_endpoints },

    };
};

auto Parameters_Method_MMF( const std::string & config_file_name ) -> toml::table
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

    return toml::table{
        { "output_file_tag", parameters.output_file_tag },
        { "output_folder", parameters.output_folder },
        { "output_any", parameters.output_any },
        { "output_initial", parameters.output_initial },
        { "output_final", parameters.output_final },
        { "output_energy_step", parameters.output_energy_step },
        { "output_energy_archive", parameters.output_energy_archive },
        { "output_energy_divide_by_nspins", parameters.output_energy_divide_by_nspins },
        { "output_energy_add_readability_lines", parameters.output_energy_add_readability_lines },
        { "output_configuration_step", parameters.output_configuration_step },
        { "output_configuration_archive", parameters.output_configuration_archive },
        { "output_configuration_filetype", output_configuration_filetype },
        // Method parameters
        { "max_walltime", str_max_walltime },
        { "force_convergence", parameters.force_convergence },
        { "n_iterations", parameters.n_iterations },
        { "n_iterations_log", parameters.n_iterations_log },
        { "n_iterations_amortize", parameters.n_iterations_amortize },
        { "n_modes", parameters.n_modes },
        { "n_mode_follow", parameters.n_mode_follow },
    };
};

auto Parameters_Method_LLG( const std::string & config_file_name ) -> toml::table
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

    return toml::table{
        { "output_file_tag", parameters.output_file_tag },
        { "output_folder", parameters.output_folder },
        { "output_any", parameters.output_any },
        { "output_initial", parameters.output_initial },
        { "output_final", parameters.output_final },
        { "output_energy_spin_resolved", parameters.output_energy_spin_resolved },
        { "output_energy_step", parameters.output_energy_step },
        { "output_energy_archive", parameters.output_energy_archive },
        { "output_energy_divide_by_nspins", parameters.output_energy_divide_by_nspins },
        { "output_energy_add_readability_lines", parameters.output_energy_add_readability_lines },
        { "output_configuration_step", parameters.output_configuration_step },
        { "output_configuration_archive", parameters.output_configuration_archive },
        { "output_configuration_filetype", output_configuration_filetype },
        // Method parameters
        { "max_walltime", str_max_walltime },
        { "seed", parameters.rng_seed },
        { "n_iterations", parameters.n_iterations },
        { "n_iterations_log", parameters.n_iterations_log },
        { "n_iterations_amortize", parameters.n_iterations_amortize },
        { "dt", parameters.dt },
        { "temperature", parameters.temperature },
        { "temperature_gradient_direction", toml_array_from_container( parameters.temperature_gradient_direction ) },
        { "temperature_gradient_inclination", parameters.temperature_gradient_inclination },
        { "damping", parameters.damping },
        { "beta", parameters.beta },
        // config_file_handle.Read_Single(parameters.renorm_sd, "llg_renorm");
        { "stt_use_gradient", parameters.stt_use_gradient },
        { "stt_magnitude", parameters.stt_magnitude },
        { "stt_polarisation_normal", toml_array_from_container( parameters.stt_polarisation_normal ) },
        { "force_convergence", parameters.force_convergence },
    };
};

auto Parameters_Method_MC( const std::string & config_file_name ) -> toml::table
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
    return toml::table{
        { "output_file_tag", parameters.output_file_tag },
        { "output_folder", parameters.output_folder },
        { "output_any", parameters.output_any },
        { "output_initial", parameters.output_initial },
        { "output_final", parameters.output_final },
        { "output_energy_spin_resolved", parameters.output_energy_spin_resolved },
        { "output_energy_step", parameters.output_energy_step },
        { "output_energy_archive", parameters.output_energy_archive },
        { "output_energy_divide_by_nspins", parameters.output_energy_divide_by_nspins },
        { "output_energy_add_readability_lines", parameters.output_energy_add_readability_lines },
        { "output_configuration_step", parameters.output_configuration_step },
        { "output_configuration_archive", parameters.output_configuration_archive },
        { "output_configuration_filetype", output_configuration_filetype },
        // Method parameters
        { "max_walltime", str_max_walltime },
        { "seed", parameters.rng_seed },
        { "n_iterations", parameters.n_iterations },
        { "n_iterations_log", parameters.n_iterations_log },
        { "n_iterations_amortize", parameters.n_iterations_amortize },
        { "temperature", parameters.temperature },
        { "metropolis_step", metropolis_step },
        // Metropolis method parameters
        { "metropolis_use_adaptive_cone", parameters.metropolis_cone_adaptive },
        { "acceptance_ratio", parameters.acceptance_ratio_target },
        { "metropolis_cone_angle", parameters.metropolis_cone_angle },
        { "metropolis_random_sample", parameters.metropolis_random_sample },
    };
};

} // namespace convert

} // namespace IO
