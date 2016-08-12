#include <Parameters_MMF.h>

namespace Data
{
	Parameters_MMF::Parameters_MMF(std::string output_folder, double force_convergence, int n_iterations, int log_steps)
    {
        this->n_iterations = n_iterations;
		this->log_steps = log_steps;
		this->force_convergence = force_convergence;
		this->output_folder = output_folder;
    }
}