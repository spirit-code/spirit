#include <io/Configparser.hpp>
#include <io/Filter_File_Handle.hpp>
#include <utility/Logging.hpp>
#include <utility/Timing.hpp>

#include <fmt/format.h>

#include <string>

using Utility::Log_Level;
using Utility::Log_Sender;

namespace IO
{

namespace
{

std::string missing_section_message( std::string_view name )
{
    return fmt::format( "Missing config section: '{}'. Using defaults...", name );
}

} // namespace

auto Parameters_Method_EMA_from_TOML( const toml::table & node ) -> std::unique_ptr<Data::Parameters_Method_EMA>
{
    static constexpr std::string_view config_path = "method.ema";

    Log( Log_Level::Debug, Log_Sender::IO, "Parameters EMA: building" );

    // Default parameters
    auto parameters = std::make_unique<Data::Parameters_Method_EMA>();

    // Maximum wall time
    std::string str_max_walltime = "0";

    if( const auto * tbl_view = node.at_path( config_path ).as_table() )
    {
        const auto & tbl = *tbl_view;

        // Output parameters
        if( auto output_table = tbl["output"].as_table() )
        {
            auto & output = *output_table;
            read_single( parameters->output_folder, output["folder"] );
            read_single( parameters->output_file_tag, output["file_tag"] );
            read_single( parameters->output_any, output["any"] );
            read_single( parameters->output_initial, output["initial"] );
            read_single( parameters->output_final, output["final"] );
            read_single( parameters->output_energy_divide_by_nspins, output["energy_divide_by_nspins"] );
            read_single( parameters->output_energy_spin_resolved, output["energy_spin_resolved"] );
            read_single( parameters->output_energy_step, output["energy_step"] );
            read_single( parameters->output_energy_archive, output["energy_archive"] );
            read_single( parameters->output_configuration_step, output["configuration_step"] );
            read_single( parameters->output_configuration_archive, output["configuration_archive"] );
        }
        // Method parameters
        read_single( str_max_walltime, tbl["max_walltime"] );
        parameters->max_walltime_sec
            = static_cast<long int>( Utility::Timing::DurationFromString( str_max_walltime ).count() );
        read_single( parameters->n_iterations, tbl["n_iterations"] );
        read_single( parameters->n_iterations_log, tbl["n_iterations_log"] );
        read_single( parameters->n_modes, tbl["n_modes"] );
        read_single( parameters->n_mode_follow, tbl["n_mode_follow"] );
        read_single( parameters->frequency, tbl["frequency"] );
        read_single( parameters->amplitude, tbl["amplitude"] );
        read_single( parameters->sparse, tbl["sparse"] );
    }
    else
    {
        Log( Log_Level::Warning, Log_Sender::IO, missing_section_message( config_path ) );
    }

    // Return
    std::vector<std::string> parameter_log;
    parameter_log.emplace_back( "Parameters EMA:" );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "n_modes", parameters->n_modes ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "n_mode_follow", parameters->n_mode_follow ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "frequency", parameters->frequency ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "amplitude", parameters->amplitude ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "n_iterations_log", parameters->n_iterations_log ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "n_iterations", parameters->n_iterations ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "sparse", parameters->sparse ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "maximum walltime", str_max_walltime ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output_configuration_archive", parameters->output_configuration_archive ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output_configuration_step", parameters->output_configuration_step ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output_energy_archive", parameters->output_energy_archive ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output_energy_step", parameters->output_energy_step ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output_energy_spin_resolved", parameters->output_energy_spin_resolved ) );
    parameter_log.emplace_back( fmt::format(
        "    {:<30} = {}", "output_energy_divide_by_nspins", parameters->output_energy_divide_by_nspins ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "output_final", parameters->output_final ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "output_initial", parameters->output_initial ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "output_any", parameters->output_any ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = \"{}\"", "output_folder", parameters->output_folder ) );
    Log( Log_Level::Parameter, Log_Sender::IO, parameter_log );

    Log( Log_Level::Debug, Log_Sender::IO, "Parameters EMA: built" );
    return parameters;
}

auto Parameters_Method_GNEB_from_TOML( const toml::table & node ) -> std::unique_ptr<Data::Parameters_Method_GNEB>
{
    static constexpr std::string_view config_path = "method.gneb";

    // Default parameters
    auto parameters = std::make_unique<Data::Parameters_Method_GNEB>();

    // Maximum wall time
    std::string str_max_walltime = "0";

    if( const auto * tbl_view = node.at_path( config_path ).as_table() )
    {
        const auto & tbl = *tbl_view;

        // Chain output filetype
        int output_chain_filetype = static_cast<int>( parameters->output_vf_filetype );

        // Parse
        Log( Log_Level::Debug, Log_Sender::IO, "Parameters GNEB: building" );
        // Output parameters
        if( auto output_table = tbl["output"].as_table() )
        {
            auto & output = *output_table;
            read_single( parameters->output_file_tag, output["file_tag"] );
            read_single( parameters->output_folder, output["folder"] );
            read_single( parameters->output_any, output["any"] );
            read_single( parameters->output_initial, output["initial"] );
            read_single( parameters->output_final, output["final"] );
            read_single( parameters->output_energies_step, output["energies_step"] );
            read_single( parameters->output_energies_add_readability_lines, output["energies_add_readability_lines"] );
            read_single( parameters->output_energies_interpolated, output["energies_interpolated"] );
            read_single( parameters->output_energies_divide_by_nspins, output["energies_divide_by_nspins"] );
            read_single( parameters->output_chain_step, output["chain_step"] );
            read_single( output_chain_filetype, output["chain_filetype"] );
        }
        parameters->output_vf_filetype = IO::VF_FileFormat( output_chain_filetype );
        // Method parameters
        read_single( str_max_walltime, tbl["max_walltime"] );
        parameters->max_walltime_sec
            = static_cast<long int>( Utility::Timing::DurationFromString( str_max_walltime ).count() );
        read_single( parameters->spring_constant, tbl["spring_constant"] );
        read_single( parameters->force_convergence, tbl["force_convergence"] );
        read_single( parameters->n_iterations, tbl["n_iterations"] );
        read_single( parameters->n_iterations_log, tbl["n_iterations_log"] );
        read_single( parameters->n_iterations_amortize, tbl["n_iterations_amortize"] );
        read_single( parameters->n_E_interpolations, tbl["n_energy_interpolations"] );
        read_single( parameters->moving_endpoints, tbl["moving_endpoints"] );
        read_single( parameters->equilibrium_delta_Rx_left, tbl["equilibrium_delta_Rx_left"] );
        read_single( parameters->equilibrium_delta_Rx_right, tbl["equilibrium_delta_Rx_right"] );
        read_single( parameters->translating_endpoints, tbl["translating_endpoints"] );
    }
    else
    {
        Log( Log_Level::Warning, Log_Sender::IO, missing_section_message( config_path ) );
    }

    // Return
    std::vector<std::string> parameter_log;
    parameter_log.emplace_back( "Parameters GNEB:" );
    parameter_log.emplace_back( fmt::format( "    {:<18} = {}", "spring_constant", parameters->spring_constant ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<18} = {}", "n_E_interpolations", parameters->n_E_interpolations ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<18} = {:e}", "force convergence", parameters->force_convergence ) );
    parameter_log.emplace_back( fmt::format( "    {:<18} = {}", "maximum walltime", str_max_walltime ) );
    parameter_log.emplace_back( fmt::format( "    {:<18} = {}", "n_iterations", parameters->n_iterations ) );
    parameter_log.emplace_back( fmt::format( "    {:<18} = {}", "n_iterations_log", parameters->n_iterations_log ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<18} = {}", "n_iterations_amortize", parameters->n_iterations_amortize ) );
    parameter_log.emplace_back( fmt::format( "    {:<18} = {}", "moving_endpoints", parameters->moving_endpoints ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<18} = {}", "equilibrium_delta_Rx_left", parameters->equilibrium_delta_Rx_left ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<18} = {}", "equilibrium_delta_Rx_right", parameters->equilibrium_delta_Rx_right ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<18} = {}", "translating_endpoints", parameters->translating_endpoints ) );
    parameter_log.emplace_back( fmt::format( "    {:<18} = \"{}\"", "output_folder", parameters->output_folder ) );
    parameter_log.emplace_back( fmt::format( "    {:<18} = {}", "output_any", parameters->output_any ) );
    parameter_log.emplace_back( fmt::format( "    {:<18} = {}", "output_initial", parameters->output_initial ) );
    parameter_log.emplace_back( fmt::format( "    {:<18} = {}", "output_final", parameters->output_final ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<18} = {}", "output_energies_step", parameters->output_energies_step ) );
    parameter_log.emplace_back( fmt::format(
        "    {:<18} = {}", "output_energies_add_readability_lines",
        parameters->output_energies_add_readability_lines ) );
    parameter_log.emplace_back( fmt::format( "    {:<18} = {}", "output_chain_step", parameters->output_chain_step ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<18} = {}", "output_chain_filetype", static_cast<int>( parameters->output_vf_filetype ) ) );
    Log( Log_Level::Parameter, Log_Sender::IO, parameter_log );

    Log( Log_Level::Debug, Log_Sender::IO, "Parameters GNEB: built" );
    return parameters;
} // end Parameters_Method_LLG_from_Config

auto Parameters_Method_LLG_from_TOML( const toml::table & node ) -> std::unique_ptr<Data::Parameters_Method_LLG>
{
    static constexpr std::string_view config_path = "method.llg";

    Log( Log_Level::Debug, Log_Sender::IO, "Parameters LLG: building" );

    // Default parameters
    auto parameters = std::make_unique<Data::Parameters_Method_LLG>();

    // Maximum wall time
    std::string str_max_walltime = "0";

    // PRNG Seed
    std::random_device random;
    parameters->rng_seed = random();
    parameters->prng     = std::mt19937( parameters->rng_seed );

    if( const auto * tbl_view = node.at_path( config_path ).as_table() )
    {
        const auto & tbl = *tbl_view;

        // Output parameters
        // Configuration output filetype
        if( auto output_table = tbl["output"].as_table() )
        {
            auto & output = *output_table;
            read_single( parameters->output_file_tag, output["file_tag"] );
            read_single( parameters->output_folder, output["folder"] );
            read_single( parameters->output_any, output["any"] );
            read_single( parameters->output_initial, output["initial"] );
            read_single( parameters->output_final, output["final"] );
            read_single( parameters->output_energy_spin_resolved, output["energy_spin_resolved"] );
            read_single( parameters->output_energy_step, output["energy_step"] );
            read_single( parameters->output_energy_archive, output["energy_archive"] );
            read_single( parameters->output_energy_divide_by_nspins, output["energy_divide_by_nspins"] );
            read_single( parameters->output_energy_add_readability_lines, output["energy_add_readability_lines"] );
            read_single( parameters->output_configuration_step, output["configuration_step"] );
            read_single( parameters->output_configuration_archive, output["configuration_archive"] );
            {
                int output_configuration_filetype = static_cast<int>( parameters->output_vf_filetype );
                read_single( output_configuration_filetype, output["configuration_filetype"] );
                parameters->output_vf_filetype = IO::VF_FileFormat( output_configuration_filetype );
            }
        }

        // Method parameters
        {
            read_single( str_max_walltime, tbl["max_walltime"] );
            parameters->max_walltime_sec
                = static_cast<long int>( Utility::Timing::DurationFromString( str_max_walltime ).count() );
        }

        read_single( parameters->rng_seed, tbl["seed"] );
        parameters->prng = std::mt19937( parameters->rng_seed );

        read_single( parameters->n_iterations, tbl["n_iterations"] );
        read_single( parameters->n_iterations_log, tbl["n_iterations_log"] );
        read_single( parameters->n_iterations_amortize, tbl["n_iterations_amortize"] );
        read_single( parameters->dt, tbl["dt"] );
        read_single( parameters->temperature, tbl["temperature"] );

        read_Vector3( parameters->temperature_gradient_direction, tbl["llg_temperature_gradient_direction"] );
        parameters->temperature_gradient_direction.normalize();

        read_single( parameters->temperature_gradient_inclination, tbl["temperature_gradient_inclination"] );
        read_single( parameters->damping, tbl["damping"] );
        read_single( parameters->beta, tbl["beta"] );
        read_single( parameters->stt_use_gradient, tbl["stt_use_gradient"] );
        read_single( parameters->stt_magnitude, tbl["stt_magnitude"] );
        read_Vector3( parameters->stt_polarisation_normal, tbl["llg_stt_polarisation_normal"] );
        parameters->stt_polarisation_normal.normalize();
        read_single( parameters->force_convergence, tbl["force_convergence"] );
    }
    else
    {
        Log( Log_Level::Warning, Log_Sender::IO, missing_section_message( config_path ) );
    }

    // Return
    std::vector<std::string> parameter_log;
    parameter_log.emplace_back( "Parameters LLG:" );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "seed", parameters->rng_seed ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "time step [ps]", parameters->dt ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "temperature [K]", parameters->temperature ) );
    parameter_log.emplace_back( fmt::format(
        "    {:<17} = {}", "temperature gradient direction", parameters->temperature_gradient_direction.transpose() ) );
    parameter_log.emplace_back( fmt::format(
        "    {:<17} = {}", "temperature gradient inclination", parameters->temperature_gradient_inclination ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "damping", parameters->damping ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "beta", parameters->beta ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "stt use gradient", parameters->stt_use_gradient ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "stt magnitude", parameters->stt_magnitude ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<17} = {}", "stt normal", parameters->stt_polarisation_normal.transpose() ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<17} = {:e}", "force convergence", parameters->force_convergence ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "maximum walltime", str_max_walltime ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "n_iterations", parameters->n_iterations ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "n_iterations_log", parameters->n_iterations_log ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<17} = {}", "n_iterations_amortize", parameters->n_iterations_amortize ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = \"{}\"", "output_folder", parameters->output_folder ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "output_any", parameters->output_any ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "output_initial", parameters->output_initial ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "output_final", parameters->output_final ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output_energy_step", parameters->output_energy_step ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output_energy_archive", parameters->output_energy_archive ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output_energy_spin_resolved", parameters->output_energy_spin_resolved ) );
    parameter_log.emplace_back( fmt::format(
        "    {:<30} = {}", "output_energy_divide_by_nspins", parameters->output_energy_divide_by_nspins ) );
    parameter_log.emplace_back( fmt::format(
        "    {:<30} = {}", "output_energy_add_readability_lines", parameters->output_energy_add_readability_lines ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output_configuration_step", parameters->output_configuration_step ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output_configuration_archive", parameters->output_configuration_archive ) );
    parameter_log.emplace_back( fmt::format(
        "    {:<30} = {}", "output_configuration_filetype", static_cast<int>( parameters->output_vf_filetype ) ) );

    Log( Log_Level::Parameter, Log_Sender::IO, parameter_log );

    Log( Log_Level::Debug, Log_Sender::IO, "Parameters LLG: built" );
    return parameters;
} // end Parameters_Method_LLG_from_Config

auto Parameters_Method_MC_from_TOML( const toml::table & node ) -> std::unique_ptr<Data::Parameters_Method_MC>
{
    static constexpr std::string_view config_path = "method.mc";

    Log( Log_Level::Debug, Log_Sender::IO, "Parameters MC: building" );

    // Default parameters
    auto parameters = std::make_unique<Data::Parameters_Method_MC>();

    // PRNG Seed
    std::random_device random;
    parameters->rng_seed = random();
    parameters->prng     = std::mt19937( parameters->rng_seed );

    // Maximum wall time
    std::string str_max_walltime = "0";

    if( const auto * tbl_view = node.at_path( config_path ).as_table() )
    {
        const auto & tbl = *tbl_view;

        // Configuration output filetype
        int output_configuration_filetype = static_cast<int>( parameters->output_vf_filetype );

        // Output parameters
        if( auto output_table = tbl["output"].as_table() )
        {
            auto & output = *output_table;
            read_single( parameters->output_file_tag, output["file_tag"] );
            read_single( parameters->output_folder, output["folder"] );
            read_single( parameters->output_any, output["any"] );
            read_single( parameters->output_initial, output["initial"] );
            read_single( parameters->output_final, output["final"] );
            read_single( parameters->output_energy_spin_resolved, output["energy_spin_resolved"] );
            read_single( parameters->output_energy_step, output["energy_step"] );
            read_single( parameters->output_energy_archive, output["energy_archive"] );
            read_single( parameters->output_energy_divide_by_nspins, output["energy_divide_by_nspins"] );
            read_single( parameters->output_energy_add_readability_lines, output["energy_add_readability_lines"] );
            read_single( parameters->output_configuration_step, output["configuration_step"] );
            read_single( parameters->output_configuration_archive, output["configuration_archive"] );
            read_single( output_configuration_filetype, output["configuration_filetype"] );
        }
        parameters->output_vf_filetype = IO::VF_FileFormat( output_configuration_filetype );
        // Method parameters
        read_single( str_max_walltime, tbl["max_walltime"] );
        parameters->max_walltime_sec
            = static_cast<long int>( Utility::Timing::DurationFromString( str_max_walltime ).count() );
        read_single( parameters->rng_seed, tbl["seed"] );
        parameters->prng = std::mt19937( parameters->rng_seed );
        read_single( parameters->n_iterations, tbl["n_iterations"] );
        read_single( parameters->n_iterations_log, tbl["n_iterations_log"] );
        read_single( parameters->n_iterations_amortize, tbl["n_iterations_amortize"] );
        read_single( parameters->temperature, tbl["temperature"] );
        // Metropolis method parameters
        {
            // Metropolis Step variable
            std::string metropolis_step = "cone";

            read_single( metropolis_step, tbl["metropolis_step"] );
            std::transform( metropolis_step.begin(), metropolis_step.end(), metropolis_step.begin(), ::tolower );

            if( metropolis_step == "sphere" )
                parameters->metropolis_step = Data::Metropolis_Step::SPHERE;
            else if( metropolis_step == "cone" )
                parameters->metropolis_step = Data::Metropolis_Step::CONE;
            else if( metropolis_step == "semi_classical" )
                parameters->metropolis_step = Data::Metropolis_Step::SEMI_CLASSICAL;
            else
            {
                parameters->metropolis_step = Data::Metropolis_Step::CONE;
                Log( Log_Level::Warning, Log_Sender::IO,
                     fmt::format( "Metropolis step \"{}\" unknown. Using \"cone\"...", metropolis_step ) );
            }
        }
        read_single( parameters->metropolis_cone_adaptive, tbl["metropolis_use_adaptive_cone"] );
        read_single( parameters->acceptance_ratio_target, tbl["acceptance_ratio"] );
        read_single( parameters->metropolis_cone_angle, tbl["metropolis_cone_angle"] );
        read_single( parameters->metropolis_random_sample, tbl["metropolis_random_sample"] );
    }
    else
    {
        Log( Log_Level::Warning, Log_Sender::IO, missing_section_message( config_path ) );
    }

    // Return
    std::vector<std::string> parameter_log;
    parameter_log.emplace_back( "Parameters MC:" );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "seed", parameters->rng_seed ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "temperature", parameters->temperature ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<17} = {}", "metropolis_step", name( parameters->metropolis_step ) ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<17} = {}", "target_acceptance_ratio", parameters->acceptance_ratio_target ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<17} = {}", "metropolis_use_adaptive_cone", parameters->metropolis_cone_adaptive ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<17} = {}", "metropolis_cone_angle", parameters->metropolis_cone_angle ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<17} = {}", "metropolis_cone_angle", parameters->metropolis_random_sample ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "maximum walltime", str_max_walltime ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "n_iterations", parameters->n_iterations ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "n_iterations_log", parameters->n_iterations_log ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<17} = {}", "n_iterations_amortize", parameters->n_iterations_amortize ) );
    // output parameters
    parameter_log.emplace_back( fmt::format( "    {:<17} = \"{}\"", "output_folder", parameters->output_folder ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "output_any", parameters->output_any ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "output_initial", parameters->output_initial ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "output_final", parameters->output_final ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output_energy_step", parameters->output_energy_step ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output_energy_archive", parameters->output_energy_archive ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output_energy_spin_resolved", parameters->output_energy_spin_resolved ) );
    parameter_log.emplace_back( fmt::format(
        "    {:<30} = {}", "output_energy_divide_by_nspins", parameters->output_energy_divide_by_nspins ) );
    parameter_log.emplace_back( fmt::format(
        "    {:<30} = {}", "output_energy_add_readability_lines", parameters->output_energy_add_readability_lines ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output_configuration_step", parameters->output_configuration_step ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output_configuration_archive", parameters->output_configuration_archive ) );
    parameter_log.emplace_back( fmt::format(
        "    {:<30} = {}", "output_configuration_filetype", static_cast<int>( parameters->output_vf_filetype ) ) );
    Log( Log_Level::Parameter, Log_Sender::IO, parameter_log );

    Log( Log_Level::Debug, Log_Sender::IO, "Parameters MC: built" );
    return parameters;
}

auto Parameters_Method_MMF_from_TOML( const toml::table & node ) -> std::unique_ptr<Data::Parameters_Method_MMF>
{
    static constexpr std::string_view config_path = "parameters.mmf";

    Log( Log_Level::Debug, Log_Sender::IO, "Parameters MMF: building" );
    // Default parameters
    auto parameters = std::make_unique<Data::Parameters_Method_MMF>();

    // Maximum wall time
    std::string str_max_walltime = "0";

    if( const auto * tbl_view = node.at_path( config_path ).as_table() )
    {
        const auto & tbl = *tbl_view;

        // Configuration output filetype
        int output_configuration_filetype = static_cast<int>( parameters->output_vf_filetype );

        // Output parameters
        if( auto output_table = tbl["output"].as_table() )
        {
            auto & output = *output_table;
            read_single( parameters->output_file_tag, output["file_tag"] );
            read_single( parameters->output_folder, output["folder"] );
            read_single( parameters->output_any, output["any"] );
            read_single( parameters->output_initial, output["initial"] );
            read_single( parameters->output_final, output["final"] );
            read_single( parameters->output_energy_step, output["energy_step"] );
            read_single( parameters->output_energy_archive, output["energy_archive"] );
            read_single( parameters->output_energy_divide_by_nspins, output["energy_divide_by_nspins"] );
            read_single( parameters->output_energy_add_readability_lines, output["energy_add_readability_lines"] );
            read_single( parameters->output_configuration_step, output["configuration_step"] );
            read_single( parameters->output_configuration_archive, output["configuration_archive"] );
            read_single( output_configuration_filetype, output["configuration_filetype"] );
            parameters->output_vf_filetype = IO::VF_FileFormat( output_configuration_filetype );
        }
        // Method parameters
        read_single( str_max_walltime, tbl["max_walltime"] );
        parameters->max_walltime_sec
            = static_cast<long int>( Utility::Timing::DurationFromString( str_max_walltime ).count() );
        read_single( parameters->force_convergence, tbl["force_convergence"] );
        read_single( parameters->n_iterations, tbl["n_iterations"] );
        read_single( parameters->n_iterations_log, tbl["n_iterations_log"] );
        read_single( parameters->n_iterations_amortize, tbl["n_iterations_amortize"] );
        read_single( parameters->n_modes, tbl["n_modes"] );
        read_single( parameters->n_mode_follow, tbl["n_mode_follow"] );
    }
    else
    {
        Log( Log_Level::Warning, Log_Sender::IO, "Section 'method.mmf' missing: using defaults..." );
    }

    // Return
    std::vector<std::string> parameter_log;
    parameter_log.emplace_back( "Parameters MMF:" );
    parameter_log.emplace_back(
        fmt::format( "    {:<17} = {:e}", "force convergence", parameters->force_convergence ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "maximum walltime", str_max_walltime ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "n_iterations", parameters->n_iterations ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "n_iterations_log", parameters->n_iterations_log ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<17} = {}", "n_iterations_amortize", parameters->n_iterations_amortize ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = \"{}\"", "output_folder", parameters->output_folder ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "output_any", parameters->output_any ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "output_initial", parameters->output_initial ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "output_final", parameters->output_final ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output_energy_step", parameters->output_energy_step ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output_energy_archive", parameters->output_energy_archive ) );
    parameter_log.emplace_back( fmt::format(
        "    {:<30} = {}", "output_energy_divide_by_nspins", parameters->output_energy_divide_by_nspins ) );
    parameter_log.emplace_back( fmt::format(
        "    {:<30} = {}", "output_energy_add_readability_lines", parameters->output_energy_add_readability_lines ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output_configuration_step", parameters->output_configuration_step ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output_configuration_archive", parameters->output_configuration_archive ) );
    parameter_log.emplace_back( fmt::format(
        "    {:<30} = {}", "output_configuration_filetype", static_cast<int>( parameters->output_vf_filetype ) ) );
    Log( Log_Level::Parameter, Log_Sender::IO, parameter_log );

    Log( Log_Level::Debug, Log_Sender::IO, "Parameters MMF: built" );
    return parameters;
}

} // namespace IO
