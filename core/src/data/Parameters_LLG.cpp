#include <Parameters_LLG.h>

namespace Data
{
	Parameters_LLG::Parameters_LLG(std::string output_folder, int seed_i, int n_iterations_i, int log_steps_i, double temperature_i, double damping_i, double time_step_i,
		bool renorm_sd_i, double stt_magnitude_i, std::vector<double> stt_polarisation_normal_i, double force_convergence):
		seed(seed_i), temperature(temperature_i), damping(damping_i), dt(time_step_i),
		renorm_sd(renorm_sd_i), stt_magnitude(stt_magnitude_i), stt_polarisation_normal(stt_polarisation_normal_i)
	{
		this->n_iterations = n_iterations_i;
		this->log_steps = log_steps_i;
		this->force_convergence = force_convergence;
		this->output_folder = output_folder;
		prng = std::mt19937(seed);
		distribution_real = std::uniform_real_distribution<double>(0.0, 1.0);
		distribution_minus_plus_one = std::uniform_real_distribution<double>(-1.0, 1.0);
		distribution_int = std::uniform_int_distribution<int>(0, 1);
	}
}