#pragma once
#ifndef SPIRIT_CORE_ENGINE_INTERACTION_ABC_HPP
#define SPIRIT_CORE_ENGINE_INTERACTION_ABC_HPP

#include <Spirit/Hamiltonian.h>
#include <Spirit/Spirit_Defines.h>
#include <data/Geometry.hpp>
#include <engine/Backend_par.hpp>
#include <engine/Vectormath_Defines.hpp>
#include <engine/common/interaction/ABC.hpp>
#include <engine/spin/Hamiltonian_Defines.hpp>

#include <memory>
#include <optional>
#include <vector>

namespace Engine
{

namespace Spin
{

namespace Interaction
{

using Common::Interaction::triplet;

/*
 * Abstract base class that specifies the interface a spin interaction must have.
 */
class ABC : public Common::Interaction::ABC
{
public:
    using state_t = vectorfield;

    ~ABC() override                = default;
    ABC( ABC && )                  = default;
    ABC( const ABC & )             = default;
    ABC & operator=( ABC && )      = default;
    ABC & operator=( const ABC & ) = default;

    // clone method so we can duplicate std::unique_ptr objects referencing the Interaction
    virtual std::unique_ptr<ABC> clone( Common::Interaction::Owner * new_hamiltonian ) const = 0;
    /*
     * Calculate the energy gradient of a spin configuration.
     * This function uses finite differences and may thus be quite inefficient.
     */
    virtual void Gradient_FD( const vectorfield & spins, vectorfield & gradient ) final;

    /*
     * Calculate the Hessian matrix of a spin configuration.
     * This function uses finite differences and may thus be quite inefficient.
     */
    virtual void Hessian_FD( const vectorfield & spins, MatrixX & hessian ) final;

    /*
     * Calculate the Energy per spin of a spin configuration.
     */
    virtual void Energy_per_Spin( const vectorfield & spins, scalarfield & energy ) = 0;

    /*
     * Calculate the Hessian matrix of a spin configuration.
     * This function uses finite differences and may thus be quite inefficient. You should
     * override it if you want to get proper performance.
     * This function is the fallback for derived classes where it has not been overridden.
     */
    virtual void Hessian( const vectorfield & spins, MatrixX & hessian );

    /*
     */
    virtual void Sparse_Hessian( const vectorfield & spins, std::vector<triplet> & hessian ) = 0;
    [[nodiscard]] virtual std::size_t Sparse_Hessian_Size_per_Cell() const;

    /*
     * Calculate the energy gradient of a spin configuration.
     * This function uses finite differences and may thus be quite inefficient. You should
     * override it if you want to get proper performance.
     * This function is the fallback for derived classes where it has not been overridden.
     */
    virtual void Gradient( const vectorfield & spins, vectorfield & gradient );

    // Calculate the Energy of a spin configuration
    [[nodiscard]] virtual scalar Energy( const vectorfield & spins ) final;

    // Calculate the total energy for a single spin
    [[nodiscard]] virtual scalar Energy_Single_Spin( int ispin, const vectorfield & spins );

    // polynomial order in the spins, connects spin gradient and energy (optional in cases we can't assign an order)
    // having a positive order means that energy can be calculated as spin.dot(gradient)/spin_order
    [[nodiscard]] virtual std::optional<int> spin_order() const = 0;

protected:
    ABC( Common::Interaction::Owner * hamiltonian, scalarfield energy_per_spin, scalar delta = 1e-3 ) noexcept
            : Common::Interaction::ABC( hamiltonian, std::move( energy_per_spin ) ), delta( delta ){};

    // local compute buffer
    scalarfield energy_per_spin;

    scalar delta = 1e-3;

    // maybe used for the GUI
    bool enabled = true;

#if defined( SPIRIT_USE_OPENMP ) || defined( SPIRIT_USE_CUDA )
    // When parallelising (cuda or openmp), we need all neighbours per spin
    static constexpr bool use_redundant_neighbours = true;
#else
    // When running on a single thread, we can ignore redundant neighbours
    static constexpr bool use_redundant_neighbours = false;
#endif

private:
    // Interaction name as string (must be unique per interaction because interactions with the same name cannot exist
    // within the same hamiltonian at the same time)
    static constexpr std::string_view name          = "Spin::Interaction::ABC";
    static constexpr std::optional<int> spin_order_ = std::nullopt;
};

/*
 * Specialization of the CRTP base class for (pure) spin interactions. All spin interactions must inherit from this
 */
template<class Derived>
class Base : public Common::Interaction::Base<Spin::Interaction::ABC, Derived>
{
protected:
    Base( Common::Interaction::Owner * hamiltonian, scalarfield energy_per_spin, scalar delta = 1e-3 ) noexcept
            : Common::Interaction::Base<Spin::Interaction::ABC, Derived>( hamiltonian, energy_per_spin, delta ){};

public:
    std::optional<int> spin_order() const final
    {
        return Derived::spin_order_;
    };
};

} // namespace Interaction

} // namespace Spin

} // namespace Engine

#endif
