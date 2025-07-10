#include <io/Configparser.hpp>

namespace IO
{

namespace
{
std::string DurationToString( const long int seconds )
{
    auto dt_h = static_cast<int>( seconds / 3600.0 );
    auto dt_m = static_cast<int>( ( seconds - 3600 * dt_h ) / 60.0 );
    auto dt_s = static_cast<int>( seconds - 3600 * dt_h - 60 * dt_m );
    return fmt::format( "{}:{}:{}", dt_h, dt_m, dt_s );
}
} // namespace

auto Parameters_Method_LLG_to_TOML( const Data::Parameters_Method_LLG & parameters ) -> toml::table
{
    return toml::table{
        { "output",
          toml::table{
              { "file_tag", parameters.output_file_tag },
              { "folder", parameters.output_folder },
              { "any", parameters.output_any },
              { "initial", parameters.output_initial },
              { "final", parameters.output_final },
              { "energy_spin_resolved", parameters.output_energy_spin_resolved },
              { "energy_step", parameters.output_energy_step },
              { "energy_archive", parameters.output_energy_archive },
              { "energy_divide_by_nspins", parameters.output_energy_divide_by_nspins },
              { "energy_add_readability_lines", parameters.output_energy_add_readability_lines },
              { "configuration_step", parameters.output_configuration_step },
              { "configuration_archive", parameters.output_configuration_archive },
              { "configuration_filetype", static_cast<int>( parameters.output_vf_filetype ) },
          } },
        // Method parameters
        { "max_walltime", DurationToString( parameters.max_walltime_sec ) },
        { "seed", parameters.rng_seed },
        { "n_iterations", parameters.n_iterations },
        { "n_iterations_log", parameters.n_iterations_log },
        { "n_iterations_amortize", parameters.n_iterations_amortize },
        { "dt", parameters.dt },
        { "temperature", parameters.temperature },
        { "temperature_gradient",
          as_inline( toml::table{
              { "direction", toml_array_from_container( parameters.temperature_gradient_direction ) },
              { "magnitude", parameters.temperature_gradient_magnitude },
          } ) },
        { "damping", parameters.damping },
        { "beta", parameters.beta },
        { "spin_current_model",
          [model = parameters.spin_current_model]
          {
              switch( model )
              {
                  case Data::SC_Model::ORBIT_TORQUE: return "orbit_torque";
                  case Data::SC_Model::TRANSFER_TORQUE: return "transfer_torque";
                  default: return "unknown";
              }
          }() },
        { "spin_current_vector",
          as_inline( toml::table{
              { "magnitude", parameters.spin_current_vector_magnitude },
              { "direction", toml_array_from_container( parameters.spin_current_vector_direction ) },
          } ) },
        { "force_convergence", parameters.force_convergence },
    }; // namespace IO
};

auto Parameters_Method_MC_to_TOML( const Data::Parameters_Method_MC & parameters ) -> toml::table
{
    return toml::table{
        { "output",
          toml::table{
              { "file_tag", parameters.output_file_tag },
              { "folder", parameters.output_folder },
              { "any", parameters.output_any },
              { "initial", parameters.output_initial },
              { "final", parameters.output_final },
              { "energy_spin_resolved", parameters.output_energy_spin_resolved },
              { "energy_step", parameters.output_energy_step },
              { "energy_archive", parameters.output_energy_archive },
              { "energy_divide_by_nspins", parameters.output_energy_divide_by_nspins },
              { "energy_add_readability_lines", parameters.output_energy_add_readability_lines },
              { "configuration_step", parameters.output_configuration_step },
              { "configuration_archive", parameters.output_configuration_archive },
              { "configuration_filetype", static_cast<int>( parameters.output_vf_filetype ) },
          } },
        // Method parameters
        { "max_walltime", DurationToString( parameters.max_walltime_sec ) },
        { "seed", parameters.rng_seed },
        { "n_iterations", parameters.n_iterations },
        { "n_iterations_log", parameters.n_iterations_log },
        { "n_iterations_amortize", parameters.n_iterations_amortize },
        { "temperature", parameters.temperature },
        { "metropolis",
          toml::table{
              { "acceptance_ratio", parameters.acceptance_ratio_target },
              { "step",
                [step = parameters.metropolis_step]
                {
                    switch( step )
                    {
                        case Data::Metropolis_Step::SPHERE: return "sphere";
                        case Data::Metropolis_Step::CONE: return "cone";
                        case Data::Metropolis_Step::SEMI_CLASSICAL: return "semi_classical";
                        default: return "unknown";
                    }
                }() },
              // Metropolis method parameters
              { "use_adaptive_cone", parameters.metropolis_cone_adaptive },
              { "cone_angle", parameters.metropolis_cone_angle },
              { "random_sample", parameters.metropolis_random_sample },
          } },
    }; // namespace IO
};

auto Parameters_Method_GNEB_to_TOML( const Data::Parameters_Method_GNEB & parameters ) -> toml::table
{
    return toml::table{
        { "output",
          toml::table{
              { "file_tag", parameters.output_file_tag },
              { "folder", parameters.output_folder },
              { "any", parameters.output_any },
              { "initial", parameters.output_initial },
              { "final", parameters.output_final },
              { "energies_step", parameters.output_energies_step },
              { "energies_add_readability_lines", parameters.output_energies_add_readability_lines },
              { "energies_interpolated", parameters.output_energies_interpolated },
              { "energies_divide_by_nspins", parameters.output_energies_divide_by_nspins },
              { "chain_step", parameters.output_chain_step },
              { "chain_filetype", static_cast<int>( parameters.output_vf_filetype ) },
          } },
        // Method parameters
        { "max_walltime", DurationToString( parameters.max_walltime_sec ) },
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
auto Parameters_Method_EMA_to_TOML( const Data::Parameters_Method_EMA & parameters ) -> toml::table
{
    return toml::table{
        // Output parameters
        { "output",
          toml::table{
              { "file_tag", parameters.output_file_tag },
              { "folder", parameters.output_folder },
              { "any", parameters.output_any },
              { "initial", parameters.output_initial },
              { "final", parameters.output_final },
              { "energy_divide_by_nspins", parameters.output_energy_divide_by_nspins },
              { "energy_spin_resolved", parameters.output_energy_spin_resolved },
              { "energy_step", parameters.output_energy_step },
              { "energy_archive", parameters.output_energy_archive },
              { "configuration_step", parameters.output_configuration_step },
              { "configuration_archive", parameters.output_configuration_archive },
          } },
        // Method parameters
        { "max_walltime", DurationToString( parameters.max_walltime_sec ) },
        { "n_iterations", parameters.n_iterations },
        { "n_iterations_log", parameters.n_iterations_log },
        { "n_modes", parameters.n_modes },
        { "n_mode_follow", parameters.n_mode_follow },
        { "frequency", parameters.frequency },
        { "amplitude", parameters.amplitude },
        { "sparse", parameters.sparse },

    };
};

auto Parameters_Method_MMF_to_TOML( const Data::Parameters_Method_MMF & parameters ) -> toml::table
{
    return toml::table{
        { "output",
          toml::table{
              { "file_tag", parameters.output_file_tag },
              { "folder", parameters.output_folder },
              { "any", parameters.output_any },
              { "initial", parameters.output_initial },
              { "final", parameters.output_final },
              { "energy_step", parameters.output_energy_step },
              { "energy_archive", parameters.output_energy_archive },
              { "energy_divide_by_nspins", parameters.output_energy_divide_by_nspins },
              { "energy_add_readability_lines", parameters.output_energy_add_readability_lines },
              { "configuration_step", parameters.output_configuration_step },
              { "configuration_archive", parameters.output_configuration_archive },
              { "configuration_filetype", static_cast<int>( parameters.output_vf_filetype ) },
          } },
        // Method parameters
        { "max_walltime", DurationToString( parameters.max_walltime_sec ) },
        { "force_convergence", parameters.force_convergence },
        { "n_iterations", parameters.n_iterations },
        { "n_iterations_log", parameters.n_iterations_log },
        { "n_iterations_amortize", parameters.n_iterations_amortize },
        { "n_modes", parameters.n_modes },
        { "n_mode_follow", parameters.n_mode_follow },
    };
};

} // namespace IO
