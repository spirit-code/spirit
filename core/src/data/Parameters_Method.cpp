#include <data/Parameters_Method.hpp>

namespace Data
{
	Parameters_Method::Parameters_Method(std::string output_folder, std::array<bool,4> output,
			long int n_iterations, long int n_iterations_log, long int max_walltime_sec, std::shared_ptr<Pinning> pinning, scalar force_convergence) :
		output_folder(output_folder), output_tag_time(output[0]), output_any(output[1]), output_initial(output[2]), output_final(output[3]),
		n_iterations(n_iterations), n_iterations_log(n_iterations_log), max_walltime_sec(max_walltime_sec), pinning(pinning), force_convergence(force_convergence)
	{
	}
}