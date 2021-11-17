#pragma once
#ifndef SPIRIT_CORE_DATA_SPIN_SYSTEM_HPP
#define SPIRIT_CORE_DATA_SPIN_SYSTEM_HPP

#include "Spirit_Defines.h"
#include <data/Geometry.hpp>
#include <data/Parameters_Method_EMA.hpp>
#include <data/Parameters_Method_GNEB.hpp>
#include <data/Parameters_Method_LLG.hpp>
#include <data/Parameters_Method_MC.hpp>
#include <data/Parameters_Method_MMF.hpp>
#include <engine/Hamiltonian.hpp>
#include <engine/Vectormath_Defines.hpp>
#include <utility/Ordered_Lock.hpp>

#include <memory>
#include <random>

namespace Data
{

/*
Spin_System contains all setup information on one system (one set of spins, one image).
This includes: Spin positions and orientations, Neighbours, Interaction constants, System parameters
*/
class Spin_System
{
public:
    // Constructor
    Spin_System(
        std::unique_ptr<Engine::Hamiltonian> hamiltonian, std::shared_ptr<Geometry> geometry,
        std::unique_ptr<Parameters_Method_LLG> llg_params, std::unique_ptr<Parameters_Method_MC> mc_params,
        std::unique_ptr<Parameters_Method_EMA> ema_params, std::unique_ptr<Parameters_Method_MMF> mmf_params,
        bool iteration_allowed );
    // Copy constructor
    Spin_System( Spin_System const & other );
    // Assignment operator
    Spin_System & operator=( Spin_System const & other );

    // Update
    void UpdateEnergy();
    void UpdateEffectiveField();

    // For multithreading
    void Lock() noexcept;
    void Unlock() noexcept;

    // Number of spins
    int nos;
    // Eigenmodes of the system: modes[nem][dim][nos]
    std::vector<std::shared_ptr<vectorfield>> modes;
    // Eigenvalues of the system
    std::vector<scalar> eigenvalues;
    // Orientations of the spins: spins[dim][nos]
    std::shared_ptr<vectorfield> spins;
    // Spin Hamiltonian
    std::shared_ptr<Engine::Hamiltonian> hamiltonian;
    // Geometric information
    std::shared_ptr<Geometry> geometry;
    // Parameters for LLG
    std::shared_ptr<Parameters_Method_LLG> llg_parameters;
    // Parameters for MC
    std::shared_ptr<Parameters_Method_MC> mc_parameters;
    // Parameters for EMA
    std::shared_ptr<Parameters_Method_EMA> ema_parameters;
    // Parameters for MMF
    std::shared_ptr<Parameters_Method_MMF> mmf_parameters;
    // Is it allowed to iterate on this system or do a singleshot?
    bool iteration_allowed;
    bool singleshot_allowed;

    // Total energy of the spin system (to be updated from outside, i.e. SIB, GNEB, ...)
    scalar E;
    std::vector<std::pair<std::string, scalar>> E_array;
    // Mean of magnetization
    Vector3 M;
    // Total effective field of the spins [3][nos]
    vectorfield effective_field;

private:
    // FIFO mutex for thread-safety
    Utility::OrderedLock ordered_lock;
};

} // namespace Data

#endif