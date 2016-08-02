#include <Parameters_GNEB.h>

namespace Data
{
	Parameters_GNEB::Parameters_GNEB(std::string output_folder, double spring_constant, double force_convergence, int n_iterations, int log_steps, int n_E_interpolations) :
		spring_constant(spring_constant), n_E_interpolations(n_E_interpolations)
	{
		this->n_iterations = n_iterations;
		this->log_steps = log_steps;
		this->force_convergence = force_convergence;
		this->output_folder = output_folder;
	}
}