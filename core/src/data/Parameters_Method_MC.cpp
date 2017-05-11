#include <data/Parameters_Method_MC.hpp>

namespace Data
{
	Parameters_Method_MC::Parameters_Method_MC(std::string output_folder, std::array<bool, 10> output, long int n_iterations, long int n_iterations_log,
		long int max_walltime_sec, int seed_i, scalar temperature_i, scalar acceptance_ratio_i) :
		Parameters_Method(output_folder, {output[0], output[1], output[2] , output[3]}, 0, n_iterations, n_iterations_log, max_walltime_sec),
			output_energy_step(output[4]), output_energy_archive(output[5]), output_energy_spin_resolved(output[6]),
			output_energy_divide_by_nspins(output[7]), output_configuration_step(output[8]), output_configuration_archive(output[9]),
			seed(seed_i), temperature(temperature_i), acceptance_ratio(acceptance_ratio_i)
    {
			prng = std::mt19937(seed);
			distribution_real = std::uniform_real_distribution<scalar>(0.0, 1.0);
			distribution_minus_plus_one = std::uniform_real_distribution<scalar>(-1.0, 1.0);
			distribution_int = std::uniform_int_distribution<int>(0, 1);
    }
}