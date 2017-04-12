#include <data/Parameters_Method_GNEB.hpp>

namespace Data
{
	Parameters_Method_GNEB::Parameters_Method_GNEB(std::string output_folder, std::array<bool,4> output, scalar force_convergence, long int n_iterations, long int n_iterations_log, scalar spring_constant, int n_E_interpolations) :
		Parameters_Method(output_folder, {output[0], output[1], output[2]}, force_convergence, n_iterations, n_iterations_log), spring_constant(spring_constant), n_E_interpolations(n_E_interpolations),
		output_energy(output[3])
	{
	}
}