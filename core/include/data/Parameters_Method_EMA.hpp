#pragma once
#ifndef DATA_PARAMETERS_METHOD_EMA_H
#define DATA_PARAMETERS_METHOD_EMA_H

#include "Spirit_Defines.h"
#include <data/Parameters_Method.hpp>

namespace Data
{
    // EMA_Parameters contains all EMA information about the spin system
    class Parameters_Method_EMA : public Parameters_Method
    {
    public:
        Parameters_Method_EMA(std::string output_folder, std::string output_file_tag, 
            std::array<bool,9> output, long int n_iterations, long int n_iterations_log, 
            long int max_walltime_sec, std::shared_ptr<Pinning> pinning, int n_modes, 
            int n_mode_follow, scalar frequency, scalar amplitude, bool snapshot );
       
        int n_modes;
        int n_mode_follow;
        scalar frequency;
        scalar amplitude;
        bool snapshot; 

        // ----------------- Output --------------
        // Energy output settings
        bool output_energy_step;
        bool output_energy_archive;
        bool output_energy_spin_resolved;
        bool output_energy_divide_by_nspins;

        // Spin configurations output settings
        bool output_configuration_step;
        bool output_configuration_archive;
    };
}
#endif
