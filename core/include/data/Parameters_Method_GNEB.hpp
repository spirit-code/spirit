#pragma once
#ifndef SPIRIT_CORE_DATA_PARAMETERS_METHOD_GNEB_HPP
#define SPIRIT_CORE_DATA_PARAMETERS_METHOD_GNEB_HPP

#include <data/Parameters_Method_Solver.hpp>

#include <random>
#include <vector>

namespace Data
{

// LLG_Parameters contains all LLG information about the spin system
struct Parameters_Method_GNEB : Parameters_Method_Solver
{
    // Strength of springs between images
    scalar spring_constant = 1;

    // The ratio of energy to reaction coordinate in the spring force
    //      0 is Rx only, 1 is E only
    scalar spring_force_ratio = 0;

    // With which minimum norm per spin the path shortening force should be applied
    scalar path_shortening_constant = 0;

    // Number of Energy interpolations between Images
    int n_E_interpolations = 10;

    // Temperature [K]
    scalar temperature = 0;
    // Seed for RNG
    int rng_seed = 2006;
    // Mersenne twister PRNG
    std::mt19937 prng = std::mt19937( rng_seed );

    // ----------------- Output --------------
    bool output_energies_step                  = false;
    bool output_energies_divide_by_nspins      = true;
    bool output_energies_add_readability_lines = false;
    bool output_energies_interpolated          = false;
    bool output_chain_step                     = false;
};

} // namespace Data

#endif