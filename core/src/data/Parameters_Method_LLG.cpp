#include <Parameters_Method_LLG.hpp>

namespace Data
{
	Parameters_Method_LLG::Parameters_Method_LLG(std::string output_folder, double force_convergence, long int n_iterations, long int n_iterations_log,
		int seed_i, double temperature_i, double damping_i, double time_step_i,
		bool renorm_sd_i, bool save_single_configurations_i,
		double stt_magnitude_i, std::vector<double> stt_polarisation_normal_i):
		Parameters_Method(output_folder, force_convergence, n_iterations, n_iterations_log),
		seed(seed_i), temperature(temperature_i), damping(damping_i), dt(time_step_i),
		renorm_sd(renorm_sd_i), save_single_configurations(save_single_configurations_i),
		stt_magnitude(stt_magnitude_i), stt_polarisation_normal(stt_polarisation_normal_i)
	{
		prng = std::mt19937(seed);
		distribution_real = std::uniform_real_distribution<double>(0.0, 1.0);
		distribution_minus_plus_one = std::uniform_real_distribution<double>(-1.0, 1.0);
		distribution_int = std::uniform_int_distribution<int>(0, 1);
	}
}