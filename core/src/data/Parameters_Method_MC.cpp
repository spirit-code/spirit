#include <data/Parameters_Method_MC.hpp>

namespace Data
{
	Parameters_Method_MC::Parameters_Method_MC(std::string output_folder, std::array<bool,4> output, scalar force_convergence, long int n_iterations, long int n_iterations_log) :
		Parameters_Method(output_folder, {output[0], output[1], output[2]}, force_convergence, n_iterations, n_iterations_log),
    output_energy(output[3])
    {
    }
}