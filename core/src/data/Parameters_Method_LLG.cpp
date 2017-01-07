#include <data/Parameters_Method_LLG.hpp>

namespace Data
{
	Parameters_Method_LLG::Parameters_Method_LLG(std::string output_folder, std::array<bool,6> save_output, scalar force_convergence, long int n_iterations, long int n_iterations_log,
		int seed_i, scalar temperature_i, scalar damping_i, scalar time_step_i, bool renorm_sd_i,
		scalar stt_magnitude_i, Vector3 stt_polarisation_normal_i):
		Parameters_Method(output_folder, {save_output[0], save_output[1], save_output[2], save_output[3]}, force_convergence, n_iterations, n_iterations_log),
		save_output_archive(save_output[4]), save_output_single(save_output[5]),
		seed(seed_i), temperature(temperature_i), damping(damping_i), dt(time_step_i), renorm_sd(renorm_sd_i),
		stt_magnitude(stt_magnitude_i), stt_polarisation_normal(stt_polarisation_normal_i)
	{
		prng = std::mt19937(seed);
		distribution_real = std::uniform_real_distribution<scalar>(0.0, 1.0);
		distribution_minus_plus_one = std::uniform_real_distribution<scalar>(-1.0, 1.0);
		distribution_int = std::uniform_int_distribution<int>(0, 1);
	}
}