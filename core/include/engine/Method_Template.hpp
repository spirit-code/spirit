#pragma once
#ifndef METHOD_TEMPLATE_H
#define METHOD_TEMPLATE_H

#include <Method.hpp>

#include "Spirit_Defines.h"
#include <data/Parameters_Method.hpp>
#include <data/Spin_System_Chain.hpp>
#include <data/Parameters_Method.hpp>
#include <engine/Vectormath.hpp>
#include <utility/Timing.hpp>
#include <utility/Logging.hpp>
#include <utility/Constants.hpp>

#include <deque>
#include <fstream>
#include <map>

namespace Engine
{
    enum class Solver
    {
        None,
        SIB,
        Heun,
        Depondt,
        NCG,
        BFGS,
        VP
    };

    /*
        Base Class for Simulation/Calculation Methods.
        It is templated to allow a flexible choice of Solver to iterate the systems.
    */
    template<Solver solver>
    class Method_Template : public Method
    {
    public:
        // Constructor to be used in derived classes
        Method_Template(std::shared_ptr<Data::Parameters_Method> parameters, int idx_img, int idx_chain) :
            Method(parameters, idx_img, idx_chain)
        {
            std::cerr << "INIT METHOD TEMPLATE" << std::endl;
        }
        
        // Solver name as string
        virtual std::string SolverName() override;
        virtual std::string SolverFullName() override;

    protected:
        // Solver_Initialise contains the initialisations of arrays etc. for a certain solver
        void Solver_Initialise() override;
        // Solver_Iteration represents one iteration of a certain Solver
        void Solver_Iteration() override;
    
    };

    // Default implementation: do nothing
    template <> inline
    void Method_Template<Solver::None>::Solver_Initialise()
    {
    };

    // Default implementation: do nothing
    template <> inline
    void Method_Template<Solver::None>::Solver_Iteration()
    {
    };

    template <> inline
    std::string Method_Template<Solver::None>::SolverName()
    {
        return "None";
    };

    template <> inline
    std::string Method_Template<Solver::None>::SolverFullName()
    {
        return "None";
    };

    // Include headers which specialize the Solver functions
    #include <Solver_SIB.hpp>
    #include <Solver_VP.hpp>
    #include <Solver_Heun.hpp>
    #include <Solver_Depondt.hpp>
    #include <Solver_NCG.hpp>
}

#endif