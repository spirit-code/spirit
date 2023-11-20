#pragma once
#ifndef SPIRIT_CORE_ENGINE_INTERACTION_GAUSSIAN_HPP
#define SPIRIT_CORE_ENGINE_INTERACTION_GAUSSIAN_HPP

#include <engine/interaction/ABC.hpp>

#include <vector>

namespace Engine
{

namespace Interaction
{

/*
The Gaussian Hamiltonian is meant for testing purposes and demonstrations. Spins do not interact.
A set of gaussians is summed with weight-factors so as to create an arbitrary energy landscape.
E = sum_i^N a_i exp( -l_i^2(m)/(2sigma_i^2) ) where l_i(m) is the distance of m to the gaussian i,
    a_i is the gaussian amplitude and sigma_i the width
*/
class Gaussian : public Interaction::Base<Gaussian>
{
public:
    // Constructor
    Gaussian( Hamiltonian * hamiltonian, scalarfield amplitude, scalarfield width, vectorfield center ) noexcept;

    void setParameters( const scalarfield & pAmplitude, const scalarfield & pWidth, const vectorfield & pCenter )
    {
        this->amplitude   = pAmplitude;
        this->width       = pWidth;
        this->center      = pCenter;
        this->n_gaussians = amplitude.size();
        hamiltonian->onInteractionChanged();
    };
    void getParameters( scalarfield & pAmplitude, scalarfield & pWidth, vectorfield & pCenter ) const
    {
        pAmplitude = this->amplitude;
        pWidth     = this->width;
        pCenter    = this->center;
    };

    bool is_contributing() const override;

    // General Hamiltonian functions
    void Energy_per_Spin( const vectorfield & spins, scalarfield & energy ) override;
    void Hessian( const vectorfield & spins, MatrixX & hessian ) override;
    void Sparse_Hessian( const vectorfield & spins, std::vector<triplet> & hessian ) override;

    void Gradient( const vectorfield & spins, vectorfield & gradient ) override;

    // Calculate the total energy for a single spin
    scalar Energy_Single_Spin( int ispin, const vectorfield & spins ) override;

    // Calculate the total energy for a single spin and a single gaussian
    scalar Energy_Single_Spin_Single_Gaussian( int ispin, int igauss, const vectorfield & spins );

    std::size_t n_gaussians = 0;

    // Interaction name as string
    static constexpr std::string_view name          = "Gaussian";
    static constexpr std::optional<int> spin_order_ = std::nullopt;

protected:
    void updateFromGeometry( const Data::Geometry * geometry ) override;

private:
    // Parameters of the energy landscape
    scalarfield amplitude;
    scalarfield width;
    vectorfield center;
};

} // namespace Interaction

} // namespace Engine

#endif
