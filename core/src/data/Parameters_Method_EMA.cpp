#include <data/Parameters_Method_EMA.hpp>

namespace Data
{
    Parameters_Method_EMA::Parameters_Method_EMA(std::string output_folder, std::string output_file_tag,
        std::array<bool,9> output, long int n_iterations, long int n_iterations_log, 
        long int max_walltime_sec, std::shared_ptr<Pinning> pinning, int n_modes, 
        int n_mode_follow, scalar frequency, scalar amplitude ) :
        Parameters_Method(output_folder, output_file_tag, {output[0], output[1], output[2]}, n_iterations,
            n_iterations_log, max_walltime_sec, pinning, 1e-12 ), output_energy_step(output[3]),
        output_energy_archive(output[4]), output_energy_spin_resolved(output[5]),
        output_energy_divide_by_nspins(output[6]), output_configuration_step(output[7]),
        output_configuration_archive(output[8]), n_modes(n_modes), n_mode_follow(n_mode_follow),
        frequency(frequency), amplitude(amplitude)
    {
    }
} // namespace Data