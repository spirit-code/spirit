#pragma once
#ifndef SPIRIT_CORE_ENGINE_SPIN_METHOD_EMA_HPP
#define SPIRIT_CORE_ENGINE_SPIN_METHOD_EMA_HPP

#include <Spirit/Spirit_Defines.h>
#include <engine/spin/Method.hpp>

#include <data/Parameters_Method_EMA.hpp>
#include <data/Parameters_Method_LLG.hpp>

namespace Engine
{

namespace Spin
{

/*
    The Eigenmode Analysis method
*/

class Method_EMA : public Method
{
public:
    // Constructor
    Method_EMA( std::shared_ptr<system_t> system, int idx_img, int idx_chain );

    // Method name as string
    std::string_view Name() override;

    // Solver name as string
    std::string_view SolverName() override;

private:
    // Iteration does one time step of the oscillation
    // S(t) = S(0) + cos(omega * t) * direction
    void Iteration() override;

    // Save the current Step's Data: spins and energy
    void Save_Current( std::string starttime, int iteration, bool initial = false, bool final = false ) override;
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

    // Lock system in order to prevent otherwise access
    void Lock() override;
    // Unlock system to re-enable access
    void Unlock() override;
    // Check if iterations are allowed
    bool Iterations_Allowed() override;

    // System the Solver will access
    std::shared_ptr<system_t> system;

    std::shared_ptr<Data::Parameters_Method_EMA> parameters_ema;

    int counter;
    int following_mode;

    vectorfield mode;
    scalarfield angle;
    scalarfield angle_initial;
    vectorfield axis;
    vectorfield spins_initial;
};

} // namespace Spin

} // namespace Engine

#endif
