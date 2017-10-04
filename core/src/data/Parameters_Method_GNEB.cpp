#include <data/Parameters_Method_GNEB.hpp>

namespace Data
{
    Parameters_Method_GNEB::Parameters_Method_GNEB(std::string output_folder, 
            std::string output_file_tag, std::array<bool,7> output, scalar force_convergence, 
            long int n_iterations, long int n_iterations_log, long int max_walltime_sec, 
            std::shared_ptr<Pinning> pinning, scalar spring_constant, int n_E_interpolations) :
        Parameters_Method_Solver(output_folder, output_file_tag, {output[0], output[1], output[2]}, 
            n_iterations, n_iterations_log, max_walltime_sec, pinning, force_convergence, 1e-3),
        spring_constant(spring_constant), n_E_interpolations(n_E_interpolations),
        output_energies_step(output[3]), output_energies_interpolated(output[4]), 
        output_energies_divide_by_nspins(output[5]), output_chain_step(output[6]), temperature(0), 
        rng_seed(2006), prng(std::mt19937(2006))
    {
    }
}