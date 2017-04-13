#include <data/Parameters_Method_MMF.hpp>

namespace Data
{
	Parameters_Method_MMF::Parameters_Method_MMF(std::string output_folder, std::array<bool,9> output, scalar force_convergence, long int n_iterations, long int n_iterations_log) :
		Parameters_Method(output_folder, {output[0], output[1], output[2] , output[3]}, force_convergence, n_iterations, n_iterations_log),
		output_energy_step(output[4]), output_energy_archive(output[5]), output_energy_divide_by_nspins(output[6]), output_configuration_step(output[7]), output_configuration_archive(output[8])
    {
    }
}