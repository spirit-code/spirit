#pragma once
#ifndef DATA_PARAMETERS_METHOD_MC_H
#define DATA_PARAMETERS_METHOD_MC_H

#include <random>
#include <vector>

#include "Spirit_Defines.h"
#include <data/Parameters_Method.hpp>

namespace Data
{
    // LLG_Parameters contains all LLG information about the spin system
    class Parameters_Method_MC : public Parameters_Method
    {
    public:
        Parameters_Method_MC( std::string output_folder, std::string output_file_tag, 
            std::array<bool,10> output, int output_configuration_filetype,
            long int n_iterations, long int n_iterations_log, long int max_walltime_sec,
            std::shared_ptr<Pinning> pinning, int rng_seed, 
            scalar temperature, scalar acceptance_ratio_target);

        // Temperature [K]
        scalar temperature;
        // Seed for RNG
        int rng_seed;

        // Mersenne twister PRNG
        std::mt19937 prng;

        // Step acceptance ratio
        scalar acceptance_ratio_target;

        // ----------------- Output --------------
        // Energy output settings
        bool output_energy_step;
        bool output_energy_archive;
        bool output_energy_spin_resolved;
        bool output_energy_divide_by_nspins;
        bool output_energy_add_readability_lines;
        // Spin configurations output settings
        bool output_configuration_step;
        bool output_configuration_archive;
        int  output_configuration_filetype;
    };
}
#endif