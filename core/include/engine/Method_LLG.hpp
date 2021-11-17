#pragma once
#ifndef SPIRIT_CORE_ENGINE_METHOD_LLG_HPP
#define SPIRIT_CORE_ENGINE_METHOD_LLG_HPP

#include "Spirit_Defines.h"
#include <data/Parameters_Method_LLG.hpp>
#include <data/Spin_System.hpp>
#include <engine/Method_Solver.hpp>

#include <vector>

namespace Engine
{

/*
    The Landau-Lifshitz-Gilbert (LLG) method
*/
template<Solver solver>
class Method_LLG : public Method_Solver<solver>
{
public:
    // Constructor
    Method_LLG( std::shared_ptr<Data::Spin_System> system, int idx_img, int idx_chain );

    double get_simulated_time() override;

    // Method name as string
    std::string Name() override;

private:
    // Prepare random numbers for thermal fields, if needed
    void Prepare_Thermal_Field() override;
    // Calculate Forces onto Systems
    void Calculate_Force(
        const std::vector<std::shared_ptr<vectorfield>> & configurations, std::vector<vectorfield> & forces ) override;
    void Calculate_Force_Virtual(
        const std::vector<std::shared_ptr<vectorfield>> & configurations, const std::vector<vectorfield> & forces,
        std::vector<vectorfield> & forces_virtual ) override;

    // Check if the Forces are converged
    bool Converged() override;

    // Save the current Step's Data: spins and energy
    void Save_Current( std::string starttime, int iteration, bool initial = false, bool final = false ) override;
    // A hook into the Method before an Iteration of the Solver
    void Hook_Pre_Iteration() override;
    // A hook into the Method after an Iteration of the Solver
    void Hook_Post_Iteration() override;

    // Sets iteration_allowed to false for the corresponding method
    void Finalize() override;

    // Last calculated forces
    std::vector<vectorfield> Gradient;
    // Convergence parameters
    std::vector<bool> force_converged;
    // Temperature distribution
    scalarfield temperature_distribution;
    // Field for stt gradient method
    vectorfield s_c_grad;
    // Current energy
    scalar current_energy;

    // Measure of simulated time in picoseconds
    double picoseconds_passed;
};

} // namespace Engine

#endif