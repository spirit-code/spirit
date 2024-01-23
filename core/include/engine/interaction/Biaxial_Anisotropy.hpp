#pragma once
#ifndef SPIRIT_CORE_ENGINE_INTERACTION_BIAXIAL_ANISOTROPY_HPP
#define SPIRIT_CORE_ENGINE_INTERACTION_BIAXIAL_ANISOTROPY_HPP

#include <engine/interaction/ABC.hpp>

namespace Engine
{

namespace Interaction
{

class Biaxial_Anisotropy : public Interaction::Base<Biaxial_Anisotropy>
{
public:
    Biaxial_Anisotropy(
        Hamiltonian * hamiltonian, intfield indices, field<AnisotropyPolynomial> polynomials ) noexcept;

    void setParameters( const intfield & pIndices, const field<AnisotropyPolynomial> & pPolynomials )
    {
        assert( pIndices.size() == pPolynomials.size() );
        this->anisotropy_indices     = pIndices;
        this->anisotropy_polynomials = pPolynomials;
        hamiltonian->onInteractionChanged();
    };
    void getParameters( intfield & pIndices, field<AnisotropyPolynomial> & pPolynomials ) const
    {
        pIndices     = this->anisotropy_indices;
        pPolynomials = this->anisotropy_polynomials;
    };
    [[nodiscard]] std::size_t getN_Atoms() const
    {
        return anisotropy_polynomials.size();
    }

    [[nodiscard]] std::size_t getN_Terms( const std::size_t i ) const
    {
        if( i >= anisotropy_polynomials.size() )
            return 0;
        return anisotropy_polynomials[i].terms.size();
    }

    bool is_contributing() const override;

    void Energy_per_Spin( const vectorfield & spins, scalarfield & energy ) override;
    void Hessian( const vectorfield & spins, MatrixX & hessian ) override;
    void Sparse_Hessian( const vectorfield & spins, std::vector<triplet> & hessian ) override;

    std::size_t Sparse_Hessian_Size_per_Cell() const override
    {
        return anisotropy_indices.size() * 9;
    };

    void Gradient( const vectorfield & spins, vectorfield & gradient ) override;

    // Calculate the total energy for a single spin to be used in Monte Carlo.
    //      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
    scalar Energy_Single_Spin( int ispin, const vectorfield & spins ) override;

    // Interaction name as string
    static constexpr std::string_view name          = "Biaxial Anisotropy";
    static constexpr std::optional<int> spin_order_ = std::nullopt;

protected:
    void updateFromGeometry( const Data::Geometry * geometry ) override;

private:
    intfield anisotropy_indices;
    field<AnisotropyPolynomial> anisotropy_polynomials;
};

} // namespace Interaction

} // namespace Engine
#endif
