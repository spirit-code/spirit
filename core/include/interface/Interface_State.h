#pragma once
#ifndef INTERFACE_GLOBALS_H
#define INTERFACE_GLOBALS_H

#include "Spin_System_Chain.h"
#include "Optimizer.h"
#include "Solver_LLG.h"
#include "Solver_GNEB.h"
#include "Solver_MMF.h"

//  State
//    The State struct is passed around in an application to make the
//    simulation's state available.
struct State
{
    // Main data container: a chain of Spin_Systems
    //    this needs to be replaced by a collection of chains for MMF
    std::shared_ptr<Data::Spin_System_Chain> c;
    // Currently active Image
    std::shared_ptr<Data::Spin_System> active_image;
    // Info
    int nos /*Number of Spins*/, noi /*Number of Images*/, noc /*Number of Chains*/;
    int idx_active_image, idx_active_chain;
    // The solvers
    //    max. noi*noc LLG solvers
    std::vector<std::vector<std::shared_ptr<Engine::Solver_LLG>>> solvers_llg;
    //    max. noc GNEB solvers
    std::vector<std::shared_ptr<Engine::Solver_GNEB>> solvers_gneb;
    //    max. noc MMF solvers
    std::vector<std::shared_ptr<Engine::Solver_MMF>> solvers_mmf;
};

// setupState
//    Create the State and fill it with initial data
extern "C" State * setupState(const char * config_file = "");

#endif