#pragma once
#ifndef SPIRIT_CORE_DATA_PARAMETERS_METHOD_EMA_HPP
#define SPIRIT_CORE_DATA_PARAMETERS_METHOD_EMA_HPP

#include <data/Parameters_Method.hpp>

namespace Data
{

// EMA_Parameters contains all EMA information about the spin system
struct Parameters_Method_EMA : Parameters_Method
{
    int n_modes       = 10;
    int n_mode_follow = 0;
    scalar frequency  = 0.02;
    scalar amplitude  = 1;
    bool snapshot     = false;

    // ----------------- Output --------------
    // Energy output settings
    bool output_energy_step             = false;
    bool output_energy_archive          = false;
    bool output_energy_spin_resolved    = false;
    bool output_energy_divide_by_nspins = true;

    // Spin configurations output settings
    bool output_configuration_step    = false;
    bool output_configuration_archive = false;
};

} // namespace Data

#endif
