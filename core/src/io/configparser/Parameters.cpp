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

auto Defaults_from_TOML( const toml::table & root ) -> Defaults
{
    return { /*output=*/{
        /*file_tag=*/root.at_path( "defaults.output.file_tag" ).value<std::string>(),
        /*directory=*/root.at_path( "defaults.output.folder" ).value<std::string>(),
    } };
}

auto Parameters_Method_EMA_from_TOML( const toml::table & tbl, const Defaults & defaults )
    -> std::unique_ptr<Data::Parameters_Method_EMA>
{
    static constexpr std::string_view config_path = "method.ema";

    Log( Log_Level::Debug, Log_Sender::IO, "Parameters EMA: building" );

    // Default parameters
    auto parameters = std::make_unique<Data::Parameters_Method_EMA>();

    // Maximum wall time
    std::string str_max_walltime = "0";

    if( tbl.at_path( config_path ).as_table() )
    {
        constexpr auto prefix = []( const std::string_view key ) { return fmt::format( "{}.{}", config_path, key ); };

        // Output parameters
        if( tbl.at_path( prefix( "output" ) ).as_table() )
        {
            constexpr auto o_prefix
                = []( const std::string_view key ) { return fmt::format( "{}.output.{}", config_path, key ); };

            read_value_with_default( tbl, o_prefix( "folder" ), parameters->output_folder, defaults.output.directory );
            read_value_with_default(
                tbl, o_prefix( "file_tag" ), parameters->output_file_tag, defaults.output.file_tag );
            read_value( tbl, o_prefix( "any" ), parameters->output_any );
            read_value( tbl, o_prefix( "initial" ), parameters->output_initial );
            read_value( tbl, o_prefix( "final" ), parameters->output_final );
            read_value( tbl, o_prefix( "energy_divide_by_nspins" ), parameters->output_energy_divide_by_nspins );
            read_value( tbl, o_prefix( "energy_spin_resolved" ), parameters->output_energy_spin_resolved );
            read_value( tbl, o_prefix( "energy_step" ), parameters->output_energy_step );
            read_value( tbl, o_prefix( "energy_archive" ), parameters->output_energy_archive );
            read_value( tbl, o_prefix( "configuration_step" ), parameters->output_configuration_step );
            read_value( tbl, o_prefix( "configuration_archive" ), parameters->output_configuration_archive );
        }
        else
            Log( Log_Level::Warning, Log_Sender::IO, missing_section_message( prefix( "output" ) ) );

        // Method parameters
        read_value( tbl, prefix( "max_walltime" ), str_max_walltime );
        parameters->max_walltime_sec
            = static_cast<long int>( Utility::Timing::DurationFromString( str_max_walltime ).count() );
        read_value( tbl, prefix( "n_iterations" ), parameters->n_iterations );
        read_value( tbl, prefix( "n_iterations_log" ), parameters->n_iterations_log );
        read_value( tbl, prefix( "n_modes" ), parameters->n_modes );
        read_value( tbl, prefix( "n_mode_follow" ), parameters->n_mode_follow );
        read_value( tbl, prefix( "frequency" ), parameters->frequency );
        read_value( tbl, prefix( "amplitude" ), parameters->amplitude );
        read_value( tbl, prefix( "sparse" ), parameters->sparse );
    }
    else
        Log( Log_Level::Warning, Log_Sender::IO, missing_section_message( config_path ) );

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
        fmt::format( "    {:<30} = {}", "output.configuration_archive", parameters->output_configuration_archive ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output.configuration_step", parameters->output_configuration_step ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output.energy_archive", parameters->output_energy_archive ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output.energy_step", parameters->output_energy_step ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output.energy_spin_resolved", parameters->output_energy_spin_resolved ) );
    parameter_log.emplace_back( fmt::format(
        "    {:<30} = {}", "output.energy_divide_by_nspins", parameters->output_energy_divide_by_nspins ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "output.final", parameters->output_final ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "output.initial", parameters->output_initial ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "output.any", parameters->output_any ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = \"{}\"", "output.folder", parameters->output_folder ) );
    Log( Log_Level::Parameter, Log_Sender::IO, parameter_log );

    Log( Log_Level::Debug, Log_Sender::IO, "Parameters EMA: built" );
    return parameters;
}

auto Parameters_Method_GNEB_from_TOML( const toml::table & tbl, const Defaults & defaults )
    -> std::unique_ptr<Data::Parameters_Method_GNEB>
{
    static constexpr std::string_view config_path = "method.gneb";

    // Default parameters
    auto parameters = std::make_unique<Data::Parameters_Method_GNEB>();

    // Maximum wall time
    std::string str_max_walltime = "0";

    if( tbl.at_path( config_path ).as_table() )
    {
        constexpr auto prefix = []( const std::string_view key ) { return fmt::format( "{}.{}", config_path, key ); };

        // Chain output filetype
        int output_chain_filetype = static_cast<int>( parameters->output_vf_filetype );

        // Parse
        Log( Log_Level::Debug, Log_Sender::IO, "Parameters GNEB: building" );
        // Output parameters
        if( tbl.at_path( prefix( "output" ) ).as_table() )
        {
            constexpr auto o_prefix
                = []( const std::string_view key ) { return fmt::format( "{}.output.{}", config_path, key ); };

            read_value_with_default( tbl, prefix( "file_tag" ), parameters->output_file_tag, defaults.output.file_tag );
            read_value_with_default( tbl, prefix( "folder" ), parameters->output_folder, defaults.output.directory );
            read_value( tbl, o_prefix( "any" ), parameters->output_any );
            read_value( tbl, o_prefix( "initial" ), parameters->output_initial );
            read_value( tbl, o_prefix( "final" ), parameters->output_final );
            read_value( tbl, o_prefix( "energies_step" ), parameters->output_energies_step );
            read_value(
                tbl, o_prefix( "energies_add_readability_lines" ), parameters->output_energies_add_readability_lines );
            read_value( tbl, o_prefix( "energies_interpolated" ), parameters->output_energies_interpolated );
            read_value( tbl, o_prefix( "energies_divide_by_nspins" ), parameters->output_energies_divide_by_nspins );
            read_value( tbl, o_prefix( "chain_step" ), parameters->output_chain_step );
            read_value( tbl, o_prefix( "chain_filetype" ), output_chain_filetype );
        }
        else
            Log( Log_Level::Warning, Log_Sender::IO, missing_section_message( prefix( "output" ) ) );

        parameters->output_vf_filetype = IO::VF_FileFormat( output_chain_filetype );

        // Method parameters
        read_value( tbl, prefix( "max_walltime" ), str_max_walltime );
        parameters->max_walltime_sec
            = static_cast<long int>( Utility::Timing::DurationFromString( str_max_walltime ).count() );
        read_value( tbl, prefix( "spring_constant" ), parameters->spring_constant );
        read_value( tbl, prefix( "force_convergence" ), parameters->force_convergence );
        read_value( tbl, prefix( "n_iterations" ), parameters->n_iterations );
        read_value( tbl, prefix( "n_iterations_log" ), parameters->n_iterations_log );
        read_value( tbl, prefix( "n_iterations_amortize" ), parameters->n_iterations_amortize );
        read_value( tbl, prefix( "n_energy_interpolations" ), parameters->n_E_interpolations );
        read_value( tbl, prefix( "moving_endpoints" ), parameters->moving_endpoints );
        read_value( tbl, prefix( "equilibrium_delta_Rx_left" ), parameters->equilibrium_delta_Rx_left );
        read_value( tbl, prefix( "equilibrium_delta_Rx_right" ), parameters->equilibrium_delta_Rx_right );
        read_value( tbl, prefix( "translating_endpoints" ), parameters->translating_endpoints );
    }
    else
        Log( Log_Level::Warning, Log_Sender::IO, missing_section_message( config_path ) );

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
    parameter_log.emplace_back( fmt::format( "    {:<18} = \"{}\"", "output.folder", parameters->output_folder ) );
    parameter_log.emplace_back( fmt::format( "    {:<18} = {}", "output.any", parameters->output_any ) );
    parameter_log.emplace_back( fmt::format( "    {:<18} = {}", "output.initial", parameters->output_initial ) );
    parameter_log.emplace_back( fmt::format( "    {:<18} = {}", "output.final", parameters->output_final ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<18} = {}", "output.energies_step", parameters->output_energies_step ) );
    parameter_log.emplace_back( fmt::format(
        "    {:<18} = {}", "output.energies_add_readability_lines",
        parameters->output_energies_add_readability_lines ) );
    parameter_log.emplace_back( fmt::format( "    {:<18} = {}", "output.chain_step", parameters->output_chain_step ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<18} = {}", "output.chain_filetype", static_cast<int>( parameters->output_vf_filetype ) ) );
    Log( Log_Level::Parameter, Log_Sender::IO, parameter_log );

    Log( Log_Level::Debug, Log_Sender::IO, "Parameters GNEB: built" );
    return parameters;
} // end Parameters_Method_LLG_from_Config

auto Parameters_Method_LLG_from_TOML( const toml::table & tbl, const Defaults & defaults )
    -> std::unique_ptr<Data::Parameters_Method_LLG>
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

    if( tbl.at_path( config_path ).as_table() )
    {
        constexpr auto prefix = []( const std::string_view key ) { return fmt::format( "{}.{}", config_path, key ); };

        // Output parameters
        // Configuration output filetype
        if( tbl.at_path( prefix( "output" ) ).as_table() )
        {
            constexpr auto o_prefix
                = []( const std::string_view key ) { return fmt::format( "{}.output.{}", config_path, key ); };

            read_value_with_default(
                tbl, o_prefix( "file_tag" ), parameters->output_file_tag, defaults.output.file_tag );
            read_value_with_default( tbl, o_prefix( "folder" ), parameters->output_folder, defaults.output.directory );
            read_value( tbl, o_prefix( "any" ), parameters->output_any );
            read_value( tbl, o_prefix( "initial" ), parameters->output_initial );
            read_value( tbl, o_prefix( "final" ), parameters->output_final );
            read_value( tbl, o_prefix( "energy_spin_resolved" ), parameters->output_energy_spin_resolved );
            read_value( tbl, o_prefix( "energy_step" ), parameters->output_energy_step );
            read_value( tbl, o_prefix( "energy_archive" ), parameters->output_energy_archive );
            read_value( tbl, o_prefix( "energy_divide_by_nspins" ), parameters->output_energy_divide_by_nspins );
            read_value(
                tbl, o_prefix( "energy_add_readability_lines" ), parameters->output_energy_add_readability_lines );
            read_value( tbl, o_prefix( "configuration_step" ), parameters->output_configuration_step );
            read_value( tbl, o_prefix( "configuration_archive" ), parameters->output_configuration_archive );
            {
                int output_configuration_filetype = static_cast<int>( parameters->output_vf_filetype );
                read_value( tbl, o_prefix( "configuration_filetype" ), output_configuration_filetype );
                parameters->output_vf_filetype = IO::VF_FileFormat( output_configuration_filetype );
            }
        }
        else
            Log( Log_Level::Warning, Log_Sender::IO, missing_section_message( prefix( "output" ) ) );

        // Method parameters
        {
            read_value( tbl, prefix( "max_walltime" ), str_max_walltime );
            parameters->max_walltime_sec
                = static_cast<long int>( Utility::Timing::DurationFromString( str_max_walltime ).count() );
        }

        read_value( tbl, prefix( "seed" ), parameters->rng_seed );
        parameters->prng = std::mt19937( parameters->rng_seed );

        read_value( tbl, prefix( "n_iterations" ), parameters->n_iterations );
        read_value( tbl, prefix( "n_iterations_log" ), parameters->n_iterations_log );
        read_value( tbl, prefix( "n_iterations_amortize" ), parameters->n_iterations_amortize );
        read_value( tbl, prefix( "dt" ), parameters->dt );
        read_value( tbl, prefix( "temperature" ), parameters->temperature );

        read_Vector3(
            tbl, prefix( "temperature_gradient" ), parameters->temperature_gradient_magnitude,
            parameters->temperature_gradient_direction );
        read_value( tbl, prefix( "damping" ), parameters->damping );
        read_value( tbl, prefix( "beta" ), parameters->beta );
        read_enum( tbl, prefix( "spin_current_model" ), parameters->spin_current_model );
        read_Vector3(
            tbl, prefix( "spin_current_vector" ), parameters->spin_current_vector_magnitude,
            parameters->spin_current_vector_direction );
        read_value( tbl, prefix( "force_convergence" ), parameters->force_convergence );
    }
    else
        Log( Log_Level::Warning, Log_Sender::IO, missing_section_message( config_path ) );

    // Return
    std::vector<std::string> parameter_log;
    parameter_log.emplace_back( "Parameters LLG:" );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "seed", parameters->rng_seed ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "time step [ps]", parameters->dt ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "temperature [K]", parameters->temperature ) );
    parameter_log.emplace_back( fmt::format(
        "    {:<17} = {}", "temperature gradient direction", parameters->temperature_gradient_direction.transpose() ) );
    parameter_log.emplace_back( fmt::format(
        "    {:<17} = {}", "temperature gradient inclination", parameters->temperature_gradient_magnitude ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "damping", parameters->damping ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "beta", parameters->beta ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "stt model", name( parameters->spin_current_model ) ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<17} = {}", "stt magnitude", parameters->spin_current_vector_magnitude ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<17} = {}", "stt direction", parameters->spin_current_vector_direction.transpose() ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<17} = {:e}", "force convergence", parameters->force_convergence ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "maximum walltime", str_max_walltime ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "n_iterations", parameters->n_iterations ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "n_iterations_log", parameters->n_iterations_log ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<17} = {}", "n_iterations_amortize", parameters->n_iterations_amortize ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = \"{}\"", "output.folder", parameters->output_folder ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "output.any", parameters->output_any ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "output.initial", parameters->output_initial ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "output.final", parameters->output_final ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output.energy_step", parameters->output_energy_step ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output.energy_archive", parameters->output_energy_archive ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output.energy_spin_resolved", parameters->output_energy_spin_resolved ) );
    parameter_log.emplace_back( fmt::format(
        "    {:<30} = {}", "output.energy_divide_by_nspins", parameters->output_energy_divide_by_nspins ) );
    parameter_log.emplace_back( fmt::format(
        "    {:<30} = {}", "output.energy_add_readability_lines", parameters->output_energy_add_readability_lines ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output.configuration_step", parameters->output_configuration_step ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output.configuration_archive", parameters->output_configuration_archive ) );
    parameter_log.emplace_back( fmt::format(
        "    {:<30} = {}", "output.configuration_filetype", static_cast<int>( parameters->output_vf_filetype ) ) );

    Log( Log_Level::Parameter, Log_Sender::IO, parameter_log );

    Log( Log_Level::Debug, Log_Sender::IO, "Parameters LLG: built" );
    return parameters;
} // end Parameters_Method_LLG_from_Config

auto Parameters_Method_MC_from_TOML( const toml::table & node, const Defaults & defaults )
    -> std::unique_ptr<Data::Parameters_Method_MC>
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
        constexpr auto prefix = []( const std::string_view key ) { return fmt::format( "{}.{}", config_path, key ); };
        const auto & tbl      = *tbl_view;

        // Configuration output filetype
        int output_configuration_filetype = static_cast<int>( parameters->output_vf_filetype );

        // Output parameters
        if( tbl.at_path( prefix( "output" ) ).as_table() )
        {
            constexpr auto o_prefix
                = []( const std::string_view key ) { return fmt::format( "{}.output.{}", config_path, key ); };
            read_value_with_default(
                tbl, o_prefix( "file_tag" ), parameters->output_file_tag, defaults.output.file_tag );
            read_value_with_default( tbl, o_prefix( "folder" ), parameters->output_folder, defaults.output.directory );
            read_value( tbl, o_prefix( "any" ), parameters->output_any );
            read_value( tbl, o_prefix( "initial" ), parameters->output_initial );
            read_value( tbl, o_prefix( "final" ), parameters->output_final );
            read_value( tbl, o_prefix( "energy_spin_resolved" ), parameters->output_energy_spin_resolved );
            read_value( tbl, o_prefix( "energy_step" ), parameters->output_energy_step );
            read_value( tbl, o_prefix( "energy_archive" ), parameters->output_energy_archive );
            read_value( tbl, o_prefix( "energy_divide_by_nspins" ), parameters->output_energy_divide_by_nspins );
            read_value(
                tbl, o_prefix( "energy_add_readability_lines" ), parameters->output_energy_add_readability_lines );
            read_value( tbl, o_prefix( "configuration_step" ), parameters->output_configuration_step );
            read_value( tbl, o_prefix( "configuration_archive" ), parameters->output_configuration_archive );
            read_value( tbl, o_prefix( "configuration_filetype" ), output_configuration_filetype );
        }
        else
            Log( Log_Level::Warning, Log_Sender::IO, missing_section_message( prefix( "output" ) ) );
        parameters->output_vf_filetype = IO::VF_FileFormat( output_configuration_filetype );

        // Method parameters
        read_value( tbl, prefix( "max_walltime" ), str_max_walltime );
        parameters->max_walltime_sec
            = static_cast<long int>( Utility::Timing::DurationFromString( str_max_walltime ).count() );
        read_value( tbl, prefix( "seed" ), parameters->rng_seed );
        parameters->prng = std::mt19937( parameters->rng_seed );
        read_value( tbl, prefix( "n_iterations" ), parameters->n_iterations );
        read_value( tbl, prefix( "n_iterations_log" ), parameters->n_iterations_log );
        read_value( tbl, prefix( "n_iterations_amortize" ), parameters->n_iterations_amortize );
        read_value( tbl, prefix( "temperature" ), parameters->temperature );
        // Metropolis method parameters
        {
            // Metropolis Step variable
            std::string metropolis_step = "cone";

            read_value( tbl, prefix( "metropolis_step" ), metropolis_step );
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
        read_value( tbl, prefix( "metropolis_use_adaptive_cone" ), parameters->metropolis_cone_adaptive );
        read_value( tbl, prefix( "acceptance_ratio" ), parameters->acceptance_ratio_target );
        read_value( tbl, prefix( "metropolis_cone_angle" ), parameters->metropolis_cone_angle );
        read_value( tbl, prefix( "metropolis_random_sample" ), parameters->metropolis_random_sample );
    }
    else
        Log( Log_Level::Warning, Log_Sender::IO, missing_section_message( config_path ) );

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
    parameter_log.emplace_back( fmt::format( "    {:<17} = \"{}\"", "output.folder", parameters->output_folder ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "output.any", parameters->output_any ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "output.initial", parameters->output_initial ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "output.final", parameters->output_final ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output.energy_step", parameters->output_energy_step ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output.energy_archive", parameters->output_energy_archive ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output.energy_spin_resolved", parameters->output_energy_spin_resolved ) );
    parameter_log.emplace_back( fmt::format(
        "    {:<30} = {}", "output.energy_divide_by_nspins", parameters->output_energy_divide_by_nspins ) );
    parameter_log.emplace_back( fmt::format(
        "    {:<30} = {}", "output.energy_add_readability_lines", parameters->output_energy_add_readability_lines ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output.configuration_step", parameters->output_configuration_step ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output.configuration_archive", parameters->output_configuration_archive ) );
    parameter_log.emplace_back( fmt::format(
        "    {:<30} = {}", "output.configuration_filetype", static_cast<int>( parameters->output_vf_filetype ) ) );
    Log( Log_Level::Parameter, Log_Sender::IO, parameter_log );

    Log( Log_Level::Debug, Log_Sender::IO, "Parameters MC: built" );
    return parameters;
}

auto Parameters_Method_MMF_from_TOML( const toml::table & tbl, const Defaults & defaults )
    -> std::unique_ptr<Data::Parameters_Method_MMF>
{
    static constexpr std::string_view config_path = "parameters.mmf";

    Log( Log_Level::Debug, Log_Sender::IO, "Parameters MMF: building" );
    // Default parameters
    auto parameters = std::make_unique<Data::Parameters_Method_MMF>();

    // Maximum wall time
    std::string str_max_walltime = "0";

    if( tbl.at_path( config_path ).as_table() )
    {
        constexpr auto prefix = []( const std::string_view key ) { return fmt::format( "{}.{}", config_path, key ); };

        // Configuration output filetype
        int output_configuration_filetype = static_cast<int>( parameters->output_vf_filetype );

        // Output parameters
        if( tbl.at_path( prefix( "output" ) ).as_table() )
        {
            constexpr auto o_prefix
                = []( const std::string_view key ) { return fmt::format( "{}.output.{}", config_path, key ); };

            read_value_with_default(
                tbl, o_prefix( "file_tag" ), parameters->output_file_tag, defaults.output.file_tag );
            read_value_with_default( tbl, o_prefix( "folder" ), parameters->output_folder, defaults.output.directory );
            read_value( tbl, o_prefix( "any" ), parameters->output_any );
            read_value( tbl, o_prefix( "initial" ), parameters->output_initial );
            read_value( tbl, o_prefix( "final" ), parameters->output_final );
            read_value( tbl, o_prefix( "energy_step" ), parameters->output_energy_step );
            read_value( tbl, o_prefix( "energy_archive" ), parameters->output_energy_archive );
            read_value( tbl, o_prefix( "energy_divide_by_nspins" ), parameters->output_energy_divide_by_nspins );
            read_value(
                tbl, o_prefix( "energy_add_readability_lines" ), parameters->output_energy_add_readability_lines );
            read_value( tbl, o_prefix( "configuration_step" ), parameters->output_configuration_step );
            read_value( tbl, o_prefix( "configuration_archive" ), parameters->output_configuration_archive );
            read_value( tbl, o_prefix( "configuration_filetype" ), output_configuration_filetype );
            parameters->output_vf_filetype = IO::VF_FileFormat( output_configuration_filetype );
        }
        else
            Log( Log_Level::Warning, Log_Sender::IO, missing_section_message( prefix( "output" ) ) );

        // Method parameters
        read_value( tbl, prefix( "max_walltime" ), str_max_walltime );
        parameters->max_walltime_sec
            = static_cast<long int>( Utility::Timing::DurationFromString( str_max_walltime ).count() );
        read_value( tbl, prefix( "force_convergence" ), parameters->force_convergence );
        read_value( tbl, prefix( "n_iterations" ), parameters->n_iterations );
        read_value( tbl, prefix( "n_iterations_log" ), parameters->n_iterations_log );
        read_value( tbl, prefix( "n_iterations_amortize" ), parameters->n_iterations_amortize );
        read_value( tbl, prefix( "n_modes" ), parameters->n_modes );
        read_value( tbl, prefix( "n_mode_follow" ), parameters->n_mode_follow );
    }
    else
        Log( Log_Level::Warning, Log_Sender::IO, missing_section_message( config_path ) );

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
    parameter_log.emplace_back( fmt::format( "    {:<17} = \"{}\"", "output.folder", parameters->output_folder ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "output.any", parameters->output_any ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "output.initial", parameters->output_initial ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "output.final", parameters->output_final ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output.energy_step", parameters->output_energy_step ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output.energy_archive", parameters->output_energy_archive ) );
    parameter_log.emplace_back( fmt::format(
        "    {:<30} = {}", "output.energy_divide_by_nspins", parameters->output_energy_divide_by_nspins ) );
    parameter_log.emplace_back( fmt::format(
        "    {:<30} = {}", "output.energy_add_readability_lines", parameters->output_energy_add_readability_lines ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output.configuration_step", parameters->output_configuration_step ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output.configuration_archive", parameters->output_configuration_archive ) );
    parameter_log.emplace_back( fmt::format(
        "    {:<30} = {}", "output.configuration_filetype", static_cast<int>( parameters->output_vf_filetype ) ) );
    Log( Log_Level::Parameter, Log_Sender::IO, parameter_log );

    Log( Log_Level::Debug, Log_Sender::IO, "Parameters MMF: built" );
    return parameters;
}

} // namespace IO
