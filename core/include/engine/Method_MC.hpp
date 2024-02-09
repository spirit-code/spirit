#pragma once
#ifndef SPIRIT_CORE_ENGINE_METHOD_MC_HPP
#define SPIRIT_CORE_ENGINE_METHOD_MC_HPP

#include <Spirit/Spirit_Defines.h>
#include <data/Spin_System.hpp>
#include <engine/Method.hpp>

#include <vector>

namespace Engine
{

template<typename system_t>
class Method_MC;

/*
    The Monte Carlo method
*/
template<>
class Method_MC<Data::Spin_System<Engine::Hamiltonian>> : public Method
{
    using system_t = Data::Spin_System<Engine::Hamiltonian>;
public:
    // Constructor
    Method_MC( std::shared_ptr<system_t> system, int idx_img, int idx_chain );

    // Method name as string
    std::string Name() override;

private:
    // Solver_Iteration represents one iteration of a certain Solver
    void Iteration() override;

    // Metropolis iteration with adaptive cone radius
    void Metropolis( const vectorfield & spins_old, vectorfield & spins_new );

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

    std::shared_ptr<Data::Parameters_Method_MC> parameters_mc;

    // Cosine of current cone angle
    scalar cone_angle;
    int n_rejected;
    scalar acceptance_ratio_current;
    int nos_nonvacant;

    // Random vector array
    vectorfield xi;
};

} // namespace Engine

#endif
