#pragma once
#ifndef SPIRIT_CORE_ENGINE_INTERACTION_CUBIC_ANISOTROPY_HPP
#define SPIRIT_CORE_ENGINE_INTERACTION_CUBIC_ANISOTROPY_HPP

#include <engine/interaction/ABC.hpp>

namespace Engine
{

namespace Interaction
{

class Cubic_Anisotropy : public Base<Cubic_Anisotropy>
{
public:
    Cubic_Anisotropy( Hamiltonian * hamiltonian, intfield indices, scalarfield magnitudes ) noexcept;
    Cubic_Anisotropy( Hamiltonian * hamiltonian, const Data::ScalarfieldData & cubic_anisotropy ) noexcept;

    void setParameters( const intfield & pIndices, const scalarfield & pMagnitudes )
    {
        this->cubic_anisotropy_indices    = pIndices;
        this->cubic_anisotropy_magnitudes = pMagnitudes;
        hamiltonian->onInteractionChanged();
    };
    void getParameters( intfield & pIndices, scalarfield & pMagnitudes ) const
    {
        pIndices    = this->cubic_anisotropy_indices;
        pMagnitudes = this->cubic_anisotropy_magnitudes;
    };

    bool is_contributing() const override;

    void Energy_per_Spin( const vectorfield & spins, scalarfield & energy ) override;
    void Hessian( const vectorfield & spins, MatrixX & hessian ) override;
    void Sparse_Hessian( const vectorfield & spins, std::vector<triplet> & hessian ) override;

    void Gradient( const vectorfield & spins, vectorfield & gradient ) override;

    // Calculate the total energy for a single spin to be used in Monte Carlo.
    //      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
    scalar Energy_Single_Spin( int ispin, const vectorfield & spins ) override;

    // Interaction name as string
    static constexpr std::string_view name          = "Cubic Anisotropy";
    static constexpr std::optional<int> spin_order_ = std::nullopt;

protected:
    void updateFromGeometry( const Data::Geometry * geometry ) override;

private:
    intfield cubic_anisotropy_indices;
    scalarfield cubic_anisotropy_magnitudes;
};

} // namespace Interaction

} // namespace Engine
#endif
