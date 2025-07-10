#pragma once
#ifndef SPIRIT_CORE_DATA_PARAMETERS_METHOD_MC_HPP
#define SPIRIT_CORE_DATA_PARAMETERS_METHOD_MC_HPP

#include <Spirit/Parameters_MC.h>
#include <data/Parameters_Method.hpp>
#include <utility/Enum.hpp>

#include <random>

namespace Data
{

enum struct Metropolis_Step
{
    SPHERE         = MC_Metropolis_Step_Spin_Sphere,
    CONE           = MC_Metropolis_Step_Spin_Cone,
    SEMI_CLASSICAL = MC_Metropolis_Step_Spin_Semi_Classical,
};

constexpr auto name( Metropolis_Step step ) -> std::string_view
{
    switch( step )
    {
        case Metropolis_Step::SPHERE: return "sphere";
        case Metropolis_Step::CONE: return "cone";
        case Metropolis_Step::SEMI_CLASSICAL: return "semi-classical";
        default: return Utility::Enum::unknown;
    }
}

} // namespace Data

template<>
inline constexpr auto ::Utility::Enum::from_string<Data::Metropolis_Step>( std::string_view str )
    -> std::optional<Data::Metropolis_Step>
{
    if( str == "sphere" )
        return Data::Metropolis_Step::SPHERE;
    else if( str == "cone" )
        return Data::Metropolis_Step::CONE;
    else if( str == "semi_classical" )
        return Data::Metropolis_Step::SEMI_CLASSICAL;
    else
        return std::nullopt;
}

namespace Data
{

// LLG_Parameters contains all LLG information about the spin system
struct Parameters_Method_MC : public Parameters_Method
{
    // Temperature [K]
    scalar temperature = 0;
    // Seed for RNG
    int rng_seed = 2006;

    // Mersenne twister PRNG
    std::mt19937 prng = std::mt19937( rng_seed );

    // Whether to sample spins randomly or in sequence in Metropolis algorithm
    bool metropolis_random_sample = true;
    // Which trial step to use for the Metropolis algorithm: sphere, cone, ...
    Metropolis_Step metropolis_step = Metropolis_Step::CONE;
    // Whether to adapt the metropolis cone angle throughout a MC run to try to hit a target acceptance ratio
    bool metropolis_cone_adaptive = true;
    // The metropolis cone angle
    scalar metropolis_cone_angle = 30;

    // Target acceptance ratio of mc steps for adaptive cone angle
    scalar acceptance_ratio_target = 0.5;

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
