#pragma once
#ifndef SPIRIT_CORE_ENGINE_INTERACTION_ZEEMANN_HPP
#define SPIRIT_CORE_ENGINE_INTERACTION_ZEEMANN_HPP

#include <utility/Constants.hpp>
#include <engine/interaction/ABC.hpp>

namespace Engine
{

namespace Interaction
{

class Zeeman : public Interaction::Base<Zeeman>
{
public:
    Zeeman( Hamiltonian * hamiltonian, scalar magnitude, Vector3 normal ) noexcept;
    Zeeman( Hamiltonian * hamiltonian, const Data::NormalVector & external_field ) noexcept;

    void setParameters( const scalar & magnitude, const Vector3 & normal )
    {
        this->external_field_magnitude = magnitude;
        this->external_field_normal    = normal;
        hamiltonian->onInteractionChanged();
    };
    void getParameters( scalar & magnitude, Vector3 & normal ) const
    {
        magnitude = this->external_field_magnitude;
        normal    = this->external_field_normal;
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
    static constexpr std::string_view name          = "Zeeman";
    static constexpr std::optional<int> spin_order_ = 1;

protected:
    void updateFromGeometry( const Data::Geometry * geometry ) override;

private:
    // ------------ Single Spin Interactions ------------
    // External magnetic field across the sample
    scalar external_field_magnitude;
    Vector3 external_field_normal;
    // External magnetic field - for now external magnetic field is homogeneous
    // If required, an additional, inhomogeneous external field should be added
    //   scalarfield external_field_magnitudes;
    //   vectorfield external_field_normals;
};

} // namespace Interaction

} // namespace Engine
#endif
