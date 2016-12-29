#include <data/Parameters_Method.hpp>

namespace Data
{
	Parameters_Method::Parameters_Method(std::string output_folder, std::array<bool,4> save_output, scalar force_convergence, long int n_iterations, long int n_iterations_log) :
		output_folder(output_folder), save_output_any(save_output[0]), save_output_initial(save_output[1]), save_output_final(save_output[2]), save_output_energy(save_output[3]), force_convergence(force_convergence), n_iterations(n_iterations), n_iterations_log(n_iterations_log)
	{
	}
}