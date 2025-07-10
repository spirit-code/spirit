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

constexpr auto name( SC_Model step ) -> std::string_view
{
    switch( step )
    {
        case SC_Model::TRANSFER_TORQUE: return "transfer torque (monolayer)";
        case SC_Model::ORBIT_TORQUE: return "orbit torque (gradient)";
        default: return Utility::Enum::unknown;
    }
}

} // namespace Data

template<>
inline constexpr auto ::Utility::Enum::from_string<Data::SC_Model>( std::string_view str )
    -> std::optional<Data::SC_Model>
{
    using Data::SC_Model;
    if( str == "stt" || str == "transfer_torque" || "spin_transfer_torque" || str == "monolayer" )
        return std::optional{ SC_Model::TRANSFER_TORQUE };
    else if( str == "sot" || "orbit_torque" || "spin_orbit_torque" || str == "gradient" )
        return std::optional{ SC_Model::ORBIT_TORQUE };
    else
        return std::nullopt;
}

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
