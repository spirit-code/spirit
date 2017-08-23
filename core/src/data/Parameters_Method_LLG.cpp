#include <data/Parameters_Method_LLG.hpp>

namespace Data
{
    Parameters_Method_LLG::Parameters_Method_LLG(std::string output_folder, std::array<bool,10> output, scalar force_convergence, long int n_iterations, long int n_iterations_log,
            long int max_walltime_sec, std::shared_ptr<Pinning> pinning, int rng_seed, scalar temperature_i, scalar damping_i, scalar beta, scalar time_step, bool renorm_sd_i,
            bool stt_use_gradient, scalar stt_magnitude_i, Vector3 stt_polarisation_normal_i):
        Parameters_Method_Solver(output_folder, {output[0], output[1], output[2], output[3]}, n_iterations, n_iterations_log, max_walltime_sec, pinning, force_convergence, time_step),
        output_energy_step(output[4]), output_energy_archive(output[5]), output_energy_spin_resolved(output[6]),
        output_energy_divide_by_nspins(output[7]), output_configuration_step(output[8]), output_configuration_archive(output[9]),
        damping(damping_i), beta(beta),
        temperature(temperature_i), rng_seed(rng_seed), prng(std::mt19937(rng_seed)),
        stt_use_gradient(stt_use_gradient), stt_magnitude(stt_magnitude_i), stt_polarisation_normal(stt_polarisation_normal_i),
        direct_minimization(false)
    {
    }
}