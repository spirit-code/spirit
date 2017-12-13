#pragma once
#ifndef METHOD_EMA_H
#define METHOD_EMA_H

#include "Spirit_Defines.h"
#include <engine/Method_Solver.hpp>

#include <data/Parameters_Method_LLG.hpp>
#include <data/Spin_System_Chain_Collection.hpp>
#include <data/Parameters_Method_EMA.hpp>

namespace Engine
{
    /*
        The Eigenmode Analysis method
    */
    class Method_EMA : public Method
    {
    public:
        // Constructor
        Method_EMA(std::shared_ptr<Data::Spin_System> system, int idx_img, int idx_chain);

        // Method name as string
        std::string Name() override;
        
    private:
        // Iteration does one time step of the oscillation
        // S(t) = S(0) + cos(omega * t) * direction
        void Iteration() override;

        // Check if the Forces are converged
        bool Converged() override;

        // Save the current Step's Data: spins and energy
        void Save_Current(std::string starttime, int iteration, bool initial=false, 
                          bool final=false) override;
        // A hook into the Method before an Iteration of the Solver
        void Hook_Pre_Iteration() override;
        // A hook into the Method after an Iteration of the Solver
        void Hook_Post_Iteration() override;

        // Sets iteration_allowed to false for the corresponding method
        void Initialize() override;
        // Sets iteration_allowed to false for the corresponding method
        void Finalize() override;

        // Log message blocks
        void Message_Start() override;
        void Message_Step() override;
        void Message_End() override;

        std::shared_ptr<Data::Parameters_Method_EMA> parameters_ema;
        
        int steps_per_period;
        scalar timestep;
        int counter;
        
        Vector3 n_init;
        Vector3 n_iter;
        Vector3 axis;
        scalar angle;
        
    };
}

#endif