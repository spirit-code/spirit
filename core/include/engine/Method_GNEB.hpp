#pragma once
#ifndef SPIRIT_CORE_ENGINE_METHOD_GNEB_HPP
#define SPIRIT_CORE_ENGINE_METHOD_GNEB_HPP

#include <vector>

#include "Spirit_Defines.h"
#include <data/Spin_System_Chain.hpp>
#include <engine/Method_Solver.hpp>

namespace Engine
{

/*
    The geodesic nudged elastic band (GNEB) method
    // TODO: reference to Paper
*/
template<Solver solver>
class Method_GNEB : public Method_Solver<solver>
{
public:
    // Constructor
    Method_GNEB( std::shared_ptr<Data::Spin_System_Chain> chain, int idx_chain );

    // Return maximum force components of the images in the chain
    std::vector<scalar> getForceMaxAbsComponent_All() override;

    // Return maximum force components of the images in the chain
    std::vector<scalar> getTorqueMaxNorm_All() override;

    // Method name as string
    std::string Name() override;

    void Calculate_Force(
        const std::vector<std::shared_ptr<vectorfield>> & configurations,
        std::vector<vectorfield> & forces ) override; // Moved to public, because of cuda device lambda restrictions

private:
    // Calculate Forces onto Systems
    void Calculate_Force_Virtual(
        const std::vector<std::shared_ptr<vectorfield>> & configurations, const std::vector<vectorfield> & forces,
        std::vector<vectorfield> & forces_virtual ) override;

    // Check if the Forces are converged
    bool Converged() override;

    // Save the current Step's Data: images and images' energies and reaction coordinates
    void Save_Current( std::string starttime, int iteration, bool initial = false, bool final = false ) override;
    // A hook into the Method before an Iteration of the Solver
    void Hook_Pre_Iteration() override;
    // A hook into the Method after an Iteration of the Solver
    void Hook_Post_Iteration() override;

    // A helper method that calculates the interpolated energies, split up into the different energy contributions
    void Calculate_Interpolated_Energy_Contributions();

    // Sets iteration_allowed to false for the chain
    void Finalize() override;

    bool Iterations_Allowed() override;

    // Lock systems in order to prevent otherwise access
    void Lock() override;
    // Unlock systems to re-enable access
    void Unlock() override;

    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Last calculated energies
    std::vector<scalar> energies;
    // Last calculated Reaction coordinates
    std::vector<scalar> Rx;
    // Last calculated forces
    std::vector<vectorfield> F_total;
    std::vector<vectorfield> F_gradient;
    std::vector<vectorfield> F_spring;
    vectorfield f_shrink;
    // Last calculated tangents
    std::vector<vectorfield> tangents;
};

} // namespace Engine

#endif