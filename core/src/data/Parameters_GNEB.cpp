#include <Parameters_GNEB.h>

namespace Data
{
	Parameters_GNEB::Parameters_GNEB(std::string output_folder, double spring_constant, double force_convergence, int n_E_interpolations):
		spring_constant(spring_constant), n_E_interpolations(n_E_interpolations)
	{
		this->force_convergence = force_convergence;
		this->output_folder = output_folder;
	}
}