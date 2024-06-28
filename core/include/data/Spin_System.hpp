#pragma once
#ifndef SPIRIT_CORE_DATA_SPIN_SYSTEM_HPP
#define SPIRIT_CORE_DATA_SPIN_SYSTEM_HPP

#include <Spirit/Spirit_Defines.h>
#include <data/Geometry.hpp>
#include <data/Misc.hpp>
#include <data/Parameters_Method_EMA.hpp>
#include <data/Parameters_Method_GNEB.hpp>
#include <data/Parameters_Method_LLG.hpp>
#include <data/Parameters_Method_MC.hpp>
#include <data/Parameters_Method_MMF.hpp>
#include <engine/Vectormath_Defines.hpp>
#include <engine/spin/Hamiltonian.hpp>
#include <utility/Ordered_Lock.hpp>

#include <memory>
#include <optional>

namespace Data
{

/*
Spin_System contains all setup information on one system (one set of spins, one image).
This includes: Spin positions and orientations, Neighbours, Interaction constants, System parameters
*/

struct System_Energy
{
    // Total energy of the spin system (to be updated from outside, i.e. SIB, GNEB, ...)
    scalar total;
    Data::vectorlabeled<scalar> per_interaction;
    Data::vectorlabeled<scalarfield> per_interaction_per_spin;
};

struct System_Magnetization
{
    // Mean of magnetization
    Vector3 mean;
    // Total effective field of the spins [3][nos]
    vectorfield effective_field;
};

template<typename HamiltonianType>
class Spin_System
{
public:
    using Hamiltonian = HamiltonianType;

    // Constructor
    Spin_System(
        std::unique_ptr<Hamiltonian> hamiltonian, std::unique_ptr<Parameters_Method_LLG> llg_params,
        std::unique_ptr<Parameters_Method_MC> mc_params, std::unique_ptr<Parameters_Method_EMA> ema_params,
        std::unique_ptr<Parameters_Method_MMF> mmf_params, bool iteration_allowed );
    // Copy constructor
    Spin_System( Spin_System const & other );
    // Assignment operator
    Spin_System & operator=( Spin_System const & other );
    Spin_System( Spin_System && other )             = default;
    Spin_System & operator=( Spin_System && other ) = default;
    ~Spin_System()                                  = default;

    // Update
    void UpdateEnergy();
    void UpdateEffectiveField();

    // For multithreading
    void lock() noexcept;
    void unlock() noexcept;

    // Number of spins
    int nos;
    // Eigenmodes of the system: modes[nem][dim][nos]
    std::vector<std::optional<vectorfield>> modes;
    // Eigenvalues of the system
    std::vector<scalar> eigenvalues;
    // Orientations of the spins: spins[dim][nos]
    std::shared_ptr<typename Hamiltonian::state_t> state;
    // Spin Hamiltonian
    std::shared_ptr<Hamiltonian> hamiltonian;
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

    System_Energy E;
    System_Magnetization M;

private:
    // FIFO mutex for thread-safety
    Utility::OrderedLock ordered_lock;
};

template<typename Hamiltonian>
class Spin_System;

} // namespace Data

#endif
