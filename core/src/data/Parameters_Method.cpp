#include <data/Parameters_Method.hpp>

namespace Data
{
	Parameters_Method::Parameters_Method(std::string output_folder, std::array<bool,3> output, scalar force_convergence, long int n_iterations, long int n_iterations_log) :
		output_folder(output_folder), output_any(output[0]), output_initial(output[1]), output_final(output[2]), force_convergence(force_convergence), n_iterations(n_iterations), n_iterations_log(n_iterations_log)
	{
	}
}