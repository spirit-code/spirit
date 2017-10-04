#include <data/Parameters_Method_Solver.hpp>

namespace Data
{
	Parameters_Method_Solver::Parameters_Method_Solver(std::string output_folder, 
            std::string output_file_tag, std::array<bool,3> output, long int n_iterations, 
            long int n_iterations_log, long int max_walltime_sec, std::shared_ptr<Pinning> pinning,
            scalar force_convergence, scalar dt) :
        Parameters_Method(output_folder, output_file_tag, {output[0], output[1], output[2]}, 
                          n_iterations, n_iterations_log, max_walltime_sec, pinning, 
                          force_convergence ), 
        dt(dt)
        {}
}