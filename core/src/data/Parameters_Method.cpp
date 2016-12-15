#include <Parameters_Method.hpp>

namespace Data
{
	Parameters_Method::Parameters_Method(std::string output_folder, scalar force_convergence, long int n_iterations, long int n_iterations_log) :
		output_folder(output_folder), force_convergence(force_convergence), n_iterations(n_iterations), n_iterations_log(n_iterations_log)
	{
	}
}