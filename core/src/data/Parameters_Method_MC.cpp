#include <data/Parameters_Method_MC.hpp>

namespace Data
{
	Parameters_Method_MC::Parameters_Method_MC(std::string output_folder, std::array<bool, 10> output, long int n_iterations, long int n_iterations_log,
		long int max_walltime_sec, std::shared_ptr<Pinning> pinning, int rng_seed, scalar temperature_i, scalar acceptance_ratio_i) :
		Parameters_Method(output_folder, {output[0], output[1], output[2] , output[3]}, 0, n_iterations, n_iterations_log, max_walltime_sec, pinning, rng_seed),
			output_energy_step(output[4]), output_energy_archive(output[5]), output_energy_spin_resolved(output[6]),
			output_energy_divide_by_nspins(output[7]), output_configuration_step(output[8]), output_configuration_archive(output[9]),
			rng_seed(rng_seed), temperature(temperature_i), acceptance_ratio(acceptance_ratio_i)
    {
    }
}