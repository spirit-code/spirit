#pragma once
#ifndef SPIRIT_CORE_DATA_PARAMETERS_METHOD_LLG_HPP
#define SPIRIT_CORE_DATA_PARAMETERS_METHOD_LLG_HPP

#include <Spirit/Parameters_LLG.h>
#include <data/Parameters_Method_Solver.hpp>
#include <engine/Vectormath_Defines.hpp>
#include <utility/Enum.hpp>

#include <random>
#include <vector>

namespace Data
{

enum struct SC_Model
{
    TRANSFER_TORQUE = LLG_SC_Model_Transfer_Torque,
    ORBIT_TORQUE    = LLG_SC_Model_Orbit_Torque,
};

inline constexpr auto enum_table( SC_Model )
{
    using E = Utility::Enum::TableElementType<SC_Model>;
    return std::array{
        // clang-format off
        E{ SC_Model::TRANSFER_TORQUE, "transfer_torque", "Monolayer (STT)", "Pinned Monolayer Approximation (Transfer Torque)" },
        E{ SC_Model::ORBIT_TORQUE,    "orbit_torque",    "Gradient (SOT)",  "Gradient Approximation (Orbit Torque)"            },
        // clang-format on
    };
}

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
    Vector3 temperature_gradient_direction = Vector3{ 1, 0, 0 };
    scalar temperature_gradient_magnitude  = 0;

    // spin current model: gradient (SOT) or monolayer (STT) approximation
    SC_Model spin_current_model = SC_Model::ORBIT_TORQUE;
    // spin current vector magnitude (prop to injected current density)
    scalar spin_current_vector_magnitude = 0;
    // spin current vector direction the interpretation depends on spin current model
    Vector3 spin_current_vector_direction = Vector3{ 1, 0, 0 };

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
