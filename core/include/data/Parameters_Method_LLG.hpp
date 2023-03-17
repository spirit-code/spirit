#pragma once
#ifndef SPIRIT_CORE_DATA_PARAMETERS_METHOD_LLG_HPP
#define SPIRIT_CORE_DATA_PARAMETERS_METHOD_LLG_HPP

#include <data/Parameters_Method_Solver.hpp>
#include <engine/Vectormath_Defines.hpp>

#include <random>
#include <vector>

namespace Data
{

// LLG_Parameters contains all LLG information about the spin system
struct Parameters_Method_LLG : Parameters_Method_Solver
{
    // Damping
    scalar damping = 0.3;
    scalar beta    = 0;

    // Seed for RNG
    int rng_seed = 2006;
    // Mersenne twister PRNG
    std::mt19937 prng = std::mt19937( rng_seed );

    // Temperature [K]
    scalar temperature = 0;
    // Temperature gradient [K]
    Vector3 temperature_gradient_direction  = Vector3{ 1, 0, 0 };
    scalar temperature_gradient_inclination = 0;

    // - true:  use gradient approximation for STT
    // - false: use pinned monolayer approximation with current in z-direction
    bool stt_use_gradient = true;
    // Spin transfer torque parameter (prop to injected current density)
    scalar stt_magnitude = 0;
    // Spin current polarisation normal vector
    Vector3 stt_polarisation_normal = Vector3{ 1, 0, 0 };

    // Do direct minimization instead of dynamics
    bool direct_minimization = false;

    // ----------------- Output --------------
    // Energy output settings
    bool output_energy_step                  = false;
    bool output_energy_archive               = false;
    bool output_energy_spin_resolved         = false;
    bool output_energy_divide_by_nspins      = true;
    bool output_energy_add_readability_lines = false;
    // Spin configurations output settings
    bool output_configuration_step    = false;
    bool output_configuration_archive = false;
};

} // namespace Data

#endif