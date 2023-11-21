#pragma once
#ifndef SPIRIT_CORE_ENGINE_INTERACTION_EXCHANGE_HPP
#define SPIRIT_CORE_ENGINE_INTERACTION_EXCHANGE_HPP

#include <engine/interaction/ABC.hpp>

namespace Engine
{

namespace Interaction
{

class Exchange : public Interaction::Base<Exchange>
{
public:
    // pairs initializer
    Exchange( Hamiltonian * hamiltonian, pairfield pairs, scalarfield magnitudes ) noexcept;
    Exchange( Hamiltonian * hamiltonian, const Data::ScalarPairfieldData & exchange ) noexcept;
    // shell initializer
    Exchange( Hamiltonian * hamiltonian, const scalarfield & shell_magnitudes ) noexcept;

    void setParameters( const pairfield & pPairs, const scalarfield & pMagnitudes )
    {
        this->exchange_shell_magnitudes = scalarfield( 0 );
        this->exchange_pairs_in         = pPairs;
        this->exchange_magnitudes_in    = pMagnitudes;
        hamiltonian->onInteractionChanged();
    };
    void getInitParameters( pairfield & pPairs, scalarfield & pMagnitudes ) const
    {
        pPairs      = this->exchange_pairs_in;
        pMagnitudes = this->exchange_magnitudes_in;
    };

    void setParameters( const scalarfield & pShellMagnitudes )
    {
        this->exchange_pairs_in         = pairfield( 0 );
        this->exchange_magnitudes_in    = scalarfield( 0 );
        this->exchange_shell_magnitudes = pShellMagnitudes;
        hamiltonian->onInteractionChanged();
    };
    void getInitParameters( scalarfield & pShellMagnitudes ) const
    {
        pShellMagnitudes = this->exchange_shell_magnitudes;
    };

    void getParameters( pairfield & pPairs, scalarfield & pMagnitudes ) const
    {
        pPairs      = this->exchange_pairs;
        pMagnitudes = this->exchange_magnitudes;
    };

    bool is_contributing() const override;

    [[nodiscard]] std::size_t getN_Pairs() const
    {
        return exchange_pairs.size();
    }

    void Energy_per_Spin( const vectorfield & spins, scalarfield & energy ) override;
    void Hessian( const vectorfield & spins, MatrixX & hessian ) override;
    void Sparse_Hessian( const vectorfield & spins, std::vector<triplet> & hessian ) override;

    std::size_t Sparse_Hessian_Size_per_Cell() const override
    {
        return exchange_pairs.size() * 2;
    };

    void Gradient( const vectorfield & spins, vectorfield & gradient ) override;

    // Calculate the total energy for a single spin to be used in Monte Carlo.
    //      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
    scalar Energy_Single_Spin( int ispin, const vectorfield & spins ) override;

    // Interaction name as string
    static constexpr std::string_view name          = "Exchange";
    static constexpr std::optional<int> spin_order_ = 2;

protected:
    void updateFromGeometry( const Data::Geometry * geometry ) override;

private:
    scalarfield exchange_shell_magnitudes;
    pairfield exchange_pairs_in;
    scalarfield exchange_magnitudes_in;
    pairfield exchange_pairs;
    scalarfield exchange_magnitudes;
};

} // namespace Interaction

} // namespace Engine
#endif
