#pragma once
#ifndef METHOD_LLG_H
#define METHOD_LLG_H

#include "Spirit_Defines.h"
#include <engine/Method_Solver.hpp>
#include <data/Spin_System.hpp>
#include <data/Parameters_Method_LLG.hpp>

#include <vector>

namespace Engine
{
    /*
        The Landau-Lifshitz-Gilbert (LLG) method
    */
    template <Solver solver>
    class Method_LLG : public Method_Solver<solver>
    {
    public:
        // Constructor
        Method_LLG(std::shared_ptr<Data::Spin_System> system, int idx_img, int idx_chain);

        // Method name as string
        std::string Name() override;

    private:
        // Calculate Forces onto Systems
        void Calculate_Force(const std::vector<std::shared_ptr<vectorfield>> & configurations, std::vector<vectorfield> & forces) override;

        // Check if the Forces are converged
        bool Converged() override;

        // Save the current Step's Data: spins and energy
        void Save_Current(std::string starttime, int iteration, bool initial=false, bool final=false) override;
        // A hook into the Optimizer before an Iteration
        void Hook_Pre_Iteration() override;
        // A hook into the Optimizer after an Iteration
        void Hook_Post_Iteration() override;

        // Sets iteration_allowed to false for the corresponding method
        void Finalize() override;

        // Last calculated forces
        std::vector<vectorfield> Gradient;
        // Convergence parameters
        std::vector<bool> force_converged;
    };
}

#endif