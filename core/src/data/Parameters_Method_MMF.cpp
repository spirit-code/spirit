#include <Parameters_Method_MMF.hpp>

namespace Data
{
	Parameters_Method_MMF::Parameters_Method_MMF(std::string output_folder, scalar force_convergence, long int n_iterations, long int n_iterations_log) :
		Parameters_Method(output_folder, force_convergence, n_iterations, n_iterations_log)
    {
    }
}