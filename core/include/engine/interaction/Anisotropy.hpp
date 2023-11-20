#pragma once
#ifndef SPIRIT_CORE_ENGINE_INTERACTION_ANISOTROPY_HPP
#define SPIRIT_CORE_ENGINE_INTERACTION_ANISOTROPY_HPP

#include <engine/interaction/ABC.hpp>

namespace Engine
{

namespace Interaction
{

class Anisotropy : public Interaction::Base<Anisotropy>
{
public:
    Anisotropy( Hamiltonian * hamiltonian, intfield indices, scalarfield magnitudes, vectorfield normals ) noexcept;
    Anisotropy( Hamiltonian * hamiltonian, const Data::VectorfieldData & anisotropy ) noexcept;

    void setParameters( const intfield & indices, const scalarfield & magnitudes, const vectorfield & normals )
    {
        this->anisotropy_indices    = indices;
        this->anisotropy_magnitudes = magnitudes;
        this->anisotropy_normals    = normals;
        hamiltonian->onInteractionChanged();
    };
    void getParameters( intfield & pIndices, scalarfield & pMagnitudes, vectorfield & pNormals ) const
    {
        pIndices    = this->anisotropy_indices;
        pMagnitudes = this->anisotropy_magnitudes;
        pNormals    = this->anisotropy_normals;
    };

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
    static constexpr std::string_view name          = "Anisotropy";
    static constexpr std::optional<int> spin_order_ = 2;

protected:
    void updateFromGeometry( const Data::Geometry * geometry ) override;

private:
    intfield anisotropy_indices;
    scalarfield anisotropy_magnitudes;
    vectorfield anisotropy_normals;
};

} // namespace Interaction

} // namespace Engine
#endif
