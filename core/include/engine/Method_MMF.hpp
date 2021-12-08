#pragma once
#ifndef SPIRIT_CORE_ENGINE_METHOD_MMF_HPP
#define SPIRIT_CORE_ENGINE_METHOD_MMF_HPP

#include "Spirit_Defines.h"
#include <data/Parameters_Method_MMF.hpp>
#include <engine/Method_Solver.hpp>

namespace Engine
{

/*
    The Minimum Mode Following (MMF) method
*/
template<Solver solver>
class Method_MMF : public Method_Solver<solver>
{
public:
    // Constructor
    Method_MMF( std::shared_ptr<Data::Spin_System> system, int idx_chain );

    // Method name as string
    std::string Name() override;

private:
    // Calculate Forces onto Systems
    void Calculate_Force(
        const std::vector<std::shared_ptr<vectorfield>> & configurations, std::vector<vectorfield> & forces ) override;

    // Check if the Forces are converged
    bool Converged() override;

    // Save the current Step's Data: images and images' energies and reaction coordinates
    void Save_Current( std::string starttime, int iteration, bool initial = false, bool final = false ) override;
    // A hook into the Method before an Iteration of the Solver
    void Hook_Pre_Iteration() override;
    // A hook into the Method after an Iteration of the Solver
    void Hook_Post_Iteration() override;

    // Sets iteration_allowed to false
    void Finalize() override;

    std::shared_ptr<Data::Spin_System> system;

    bool switched1, switched2;

    // Last calculated hessian
    MatrixX hessian;
    // Last calculated gradient
    vectorfield gradient;
    // Last calculated minimum mode
    vectorfield minimum_mode;
    int mode_follow_previous;
    VectorX mode_2N_previous;

    // Last iterations spins and reaction coordinate
    scalar Rx_last;
    vectorfield spins_last;

    // Which minimum mode function to use
    // ToDo: move into parameters
    std::string mm_function;

    // Functions for getting the minimum mode of a Hessian
    void Calculate_Force_Spectra_Matrix(
        const std::vector<std::shared_ptr<vectorfield>> & configurations, std::vector<vectorfield> & forces );
    void Calculate_Force_Lanczos(
        const std::vector<std::shared_ptr<vectorfield>> configurations, std::vector<vectorfield> & forces );
};

} // namespace Engine

#endif