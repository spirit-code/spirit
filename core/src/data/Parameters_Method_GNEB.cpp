#include <data/Parameters_Method_GNEB.hpp>

namespace Data
{
    Parameters_Method_GNEB::Parameters_Method_GNEB(std::string output_folder, 
            std::string output_file_tag, std::array<bool,8> output, int output_chain_filetype,
            scalar force_convergence, long int n_iterations, long int n_iterations_log, long int max_walltime_sec, 
            scalar spring_constant, int n_E_interpolations) :
        Parameters_Method_Solver(output_folder, output_file_tag, {output[0], output[1], output[2]}, 
            n_iterations, n_iterations_log, max_walltime_sec, force_convergence, 1e-3),
        spring_constant(spring_constant), n_E_interpolations(n_E_interpolations),
        output_energies_step(output[3]), output_energies_interpolated(output[4]), 
        output_energies_divide_by_nspins(output[5]), output_chain_step(output[6]),
        output_energies_add_readability_lines(output[7]), output_chain_filetype(output_chain_filetype),
        temperature(0), rng_seed(2006), prng(std::mt19937(2006))
    {
    }
}