#pragma once
#ifndef SPIRIT_CORE_ENGINE_INTERACTION_DMI_HPP
#define SPIRIT_CORE_ENGINE_INTERACTION_DMI_HPP

#include <engine/interaction/ABC.hpp>

namespace Engine
{

namespace Interaction
{

class DMI : public Interaction::Base<DMI>
{
public:
    DMI( Hamiltonian * hamiltonian, pairfield pairs, scalarfield magnitudes, vectorfield normals ) noexcept;
    DMI( Hamiltonian * hamiltonian, const Data::VectorPairfieldData & dmi ) noexcept;
    DMI( Hamiltonian * hamiltonian, scalarfield shell_magnitudes, int chirality ) noexcept;

    void setParameters( const pairfield & pPairs, const scalarfield & pMagnitudes, const vectorfield & pNormals )
    {
        this->dmi_shell_magnitudes = scalarfield( 0 );
        this->dmi_shell_chirality  = 0;
        this->dmi_pairs_in         = pPairs;
        this->dmi_magnitudes_in    = pMagnitudes;
        this->dmi_normals_in       = pNormals;
        hamiltonian->onInteractionChanged();
    };
    void getInitParameters( pairfield & pPairs, scalarfield & pMagnitudes, vectorfield & pNormals ) const
    {
        pPairs      = this->dmi_pairs_in;
        pMagnitudes = this->dmi_magnitudes_in;
        pNormals    = this->dmi_normals_in;
    };

    bool is_contributing() const override;

    void setParameters( const scalarfield & pShellMagnitudes, int pChirality )
    {
        this->dmi_shell_magnitudes = pShellMagnitudes;
        this->dmi_shell_chirality  = pChirality;
        this->dmi_pairs_in         = pairfield( 0 );
        this->dmi_magnitudes_in    = scalarfield( 0 );
        this->dmi_normals_in       = vectorfield( 0 );
        hamiltonian->onInteractionChanged();
    };
    void getInitParameters( scalarfield & shell_magnitudes, int & chirality ) const
    {
        shell_magnitudes = this->dmi_shell_magnitudes;
        chirality        = this->dmi_shell_chirality;
    };

    void getParameters( pairfield & pPairs, scalarfield & pMagnitudes, vectorfield & pNormals ) const
    {
        pPairs      = this->dmi_pairs;
        pMagnitudes = this->dmi_magnitudes;
        pNormals    = this->dmi_normals;
    };

    [[nodiscard]] std::size_t getN_Pairs() const
    {
        return dmi_pairs.size();
    };

    void Energy_per_Spin( const vectorfield & spins, scalarfield & energy ) override;
    void Hessian( const vectorfield & spins, MatrixX & hessian ) override;
    void Sparse_Hessian( const vectorfield & spins, std::vector<triplet> & hessian ) override;

    std::size_t Sparse_Hessian_Size_per_Cell() const override
    {
        return dmi_pairs.size() * 3;
    };

    void Gradient( const vectorfield & spins, vectorfield & gradient ) override;

    // Calculate the total energy for a single spin to be used in Monte Carlo.
    //      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
    scalar Energy_Single_Spin( int ispin, const vectorfield & spins ) override;

    // Interaction name as string
    static constexpr std::string_view name          = "DMI";
    static constexpr std::optional<int> spin_order_ = 2;

protected:
    void updateFromGeometry( const Data::Geometry * geometry ) override;

private:
    scalarfield dmi_shell_magnitudes;
    int dmi_shell_chirality;
    pairfield dmi_pairs_in;
    scalarfield dmi_magnitudes_in;
    vectorfield dmi_normals_in;
    pairfield dmi_pairs;
    scalarfield dmi_magnitudes;
    vectorfield dmi_normals;
};

} // namespace Interaction

} // namespace Engine
#endif
