#pragma once
#ifndef SPIRIT_CORE_ENGINE_HAMILTONIAN_GAUSSIAN_HPP
#define SPIRIT_CORE_ENGINE_HAMILTONIAN_GAUSSIAN_HPP

#include <vector>

#include "Spirit_Defines.h"
#include <data/Geometry.hpp>
#include <engine/Hamiltonian.hpp>
#include <engine/Vectormath_Defines.hpp>

namespace Engine
{

/*
The Gaussian Hamiltonian is meant for testing purposes and demonstrations. Spins do not interact.
A set of gaussians is summed with weight-factors so as to create an arbitrary energy landscape.
E = sum_i^N a_i exp( -l_i^2(m)/(2sigma_i^2) ) where l_i(m) is the distance of m to the gaussian i,
    a_i is the gaussian amplitude and sigma_i the width
*/
class Hamiltonian_Gaussian : public Hamiltonian
{
public:
    // Constructor
    Hamiltonian_Gaussian( std::vector<scalar> amplitude, std::vector<scalar> width, std::vector<Vector3> center );

    void Update_Energy_Contributions() override;

    // General Hamiltonian functions
    void Hessian( const vectorfield & spins, MatrixX & hessian ) override;
    void Gradient( const vectorfield & spins, vectorfield & gradient ) override;
    void Energy_Contributions_per_Spin(
        const vectorfield & spins, std::vector<std::pair<std::string, scalarfield>> & contributions ) override;

    // Calculate the total energy for a single spin
    scalar Energy_Single_Spin( int ispin, const vectorfield & spins ) override;

    // Hamiltonian name as string
    const std::string & Name() override;

    // Parameters of the energy landscape
    int n_gaussians;
    std::vector<scalar> amplitude;
    std::vector<scalar> width;
    std::vector<Vector3> center;
};

} // namespace Engine

#endif