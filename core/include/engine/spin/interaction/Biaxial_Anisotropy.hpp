#pragma once
#ifndef SPIRIT_CORE_ENGINE_INTERACTION_BIAXIAL_ANISOTROPY_HPP
#define SPIRIT_CORE_ENGINE_INTERACTION_BIAXIAL_ANISOTROPY_HPP

#include <engine/spin/interaction/ABC.hpp>

namespace Engine
{

namespace Spin
{

namespace Interaction
{

/*
 * Biaxial Anisotropy
 * The terms use a CSR like format. The site_p attribute stores the information which term corresponds to which site,
 * such that the terms for the atom at `indices[i]` are the ones between `site_p[i]` & `site_p[i+1]`.
 */
class Biaxial_Anisotropy : public Interaction::Base<Biaxial_Anisotropy>
{
public:
    Biaxial_Anisotropy(
        Common::Interaction::Owner * hamiltonian, intfield indices, field<PolynomialBasis> pBases, field<unsigned int> pSite_ptr,
        field<PolynomialTerm> pTerms ) noexcept;

    void setParameters(
        const intfield & pIndices, const field<PolynomialBasis> & pBases, const field<unsigned int> & pSite_ptr,
        const field<PolynomialTerm> & pTerms )
    {
        assert( pIndices.size() == pBases.size() );
        assert( ( pIndices.empty() && pSite_ptr.empty() ) || ( pIndices.size() + 1 == pSite_ptr.size() ) );
        assert( pSite_ptr.empty() || pSite_ptr.back() == pTerms.size() );

        this->indices = pIndices;
        this->bases   = pBases;
        this->site_p  = pSite_ptr;
        this->terms   = pTerms;
        onInteractionChanged();
    };
    void getParameters(
        intfield & pIndices, field<PolynomialBasis> & pBases, field<unsigned int> & pSite_ptr,
        field<PolynomialTerm> & pTerms ) const
    {
        pIndices  = this->indices;
        pBases    = this->bases;
        pSite_ptr = this->site_p;
        pTerms    = this->terms;
    };
    [[nodiscard]] std::size_t getN_Atoms() const
    {
        return indices.size();
    }

    [[nodiscard]] std::size_t getN_Terms() const
    {
        return terms.size();
    }

    bool is_contributing() const override;

    void Energy_per_Spin( const vectorfield & spins, scalarfield & energy ) override;
    void Hessian( const vectorfield & spins, MatrixX & hessian ) override;
    void Sparse_Hessian( const vectorfield & spins, std::vector<triplet> & hessian ) override;

    std::size_t Sparse_Hessian_Size_per_Cell() const override
    {
        return indices.size() * 9;
    };

    void Gradient( const vectorfield & spins, vectorfield & gradient ) override;

    // Calculate the total energy for a single spin to be used in Monte Carlo.
    //      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
    scalar Energy_Single_Spin( int ispin, const vectorfield & spins ) override;

    // Interaction name as string
    static constexpr std::string_view name          = "Biaxial Anisotropy";
    static constexpr std::optional<int> spin_order_ = std::nullopt;

protected:
    void updateFromGeometry( const Data::Geometry & geometry ) override;

private:
    template<typename F>
    void Hessian_Impl( const vectorfield & spins, F f );

    intfield indices;
    field<PolynomialBasis> bases;
    field<unsigned int> site_p;
    field<PolynomialTerm> terms;
};

} // namespace Interaction

} // namespace Spin

} // namespace Engine
#endif
