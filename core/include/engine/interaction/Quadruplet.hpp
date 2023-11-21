#pragma once
#ifndef SPIRIT_CORE_ENGINE_INTERACTION_QUADRUPLET_HPP
#define SPIRIT_CORE_ENGINE_INTERACTION_QUADRUPLET_HPP

#include <engine/interaction/ABC.hpp>

namespace Engine
{

namespace Interaction
{

class Quadruplet : public Interaction::Base<Quadruplet>
{
public:
    Quadruplet( Hamiltonian * hamiltonian, quadrupletfield quadruplets, scalarfield magnitudes ) noexcept;
    Quadruplet( Hamiltonian * hamiltonian, const Data::QuadrupletfieldData & quadruplet ) noexcept;

    void setParameters( const quadrupletfield & pQuadruplets, const scalarfield & pMagnitudes )
    {
        this->quadruplets           = pQuadruplets;
        this->quadruplet_magnitudes = pMagnitudes;
        hamiltonian->onInteractionChanged();
    };
    void getParameters( quadrupletfield & pQuadruplets, scalarfield & pMagnitudes ) const
    {
        pQuadruplets = this->quadruplets;
        pMagnitudes  = this->quadruplet_magnitudes;
    };

    template<typename Callable>
    void apply(Callable f);

    bool is_contributing() const override;

    void Energy_per_Spin( const vectorfield & spins, scalarfield & energy ) override;
    void Hessian( const vectorfield & spins, MatrixX & hessian ) override;
    void Sparse_Hessian( const vectorfield & spins, std::vector<triplet> & hessian ) override;

    void Gradient( const vectorfield & spins, vectorfield & gradient ) override;

    // Calculate the total energy for a single spin to be used in Monte Carlo.
    //      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
    scalar Energy_Single_Spin( int ispin, const vectorfield & spins ) override;

    // Interaction name as string
    static constexpr std::string_view name = "Quadruplet";
    static constexpr std::optional<int> spin_order_ = 4;

protected:
    void updateFromGeometry(const Data::Geometry * geometry) override;

private:
    quadrupletfield quadruplets;
    scalarfield quadruplet_magnitudes;
};

} // namespace Interaction

} // namespace Engine
#endif
