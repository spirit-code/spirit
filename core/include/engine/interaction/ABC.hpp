#pragma once
#ifndef SPIRIT_CORE_ENGINE_INTERACTION_ABC_HPP
#define SPIRIT_CORE_ENGINE_INTERACTION_ABC_HPP

#include <Spirit/Hamiltonian.h>
#include <Spirit/Spirit_Defines.h>
#include <data/Geometry.hpp>
#include <engine/Backend_par.hpp>
#include <engine/Hamiltonian_Defines.hpp>
#include <engine/Vectormath_Defines.hpp>
#include <engine/interaction/Hamiltonian.hpp>

#include <memory>
#include <optional>
#include <random>
#include <vector>

namespace Engine
{

namespace Interaction
{
void setOwnerPtr( ABC & interaction, Hamiltonian * hamiltonian ) noexcept;
/*
 * Abstract base class that specifies the interface an interaction must have.
 */
class ABC
{
    friend void setOwnerPtr( ABC & interaction, Hamiltonian * hamiltonian ) noexcept;

public:
    virtual ~ABC()                 = default;
    ABC( ABC && )                  = default;
    ABC( const ABC & )             = default;
    ABC & operator=( ABC && )      = default;
    ABC & operator=( const ABC & ) = default;

    // clone method so we can duplicate std::unique_ptr objects referencing the Interaction
    virtual std::unique_ptr<ABC> clone( Hamiltonian * new_hamiltonian ) const = 0;
    virtual void updateGeometry()                                             = 0;

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

    // Interaction name as string (must be unique per interaction because interactions with the same name cannot exist
    // within the same hamiltonian at the same time)
    [[nodiscard]] virtual std::string_view Name() const = 0;

    // polynomial order in the spins, connects spin gradient and energy (optional in cases we can't assign an order)
    // having a positive order means that energy can be calculated as spin.dot(gradient)/spin_order
    [[nodiscard]] virtual std::optional<int> spin_order() const = 0;

    [[nodiscard]] virtual bool is_contributing() const
    {
        return true;
    };

    [[nodiscard]] virtual bool is_enabled() const final
    {
        return enabled;
    };

    [[nodiscard]] virtual bool is_active() const final
    {
        return is_enabled() && is_contributing();
    };

    virtual void enable() final
    {
        this->enabled = true;
        hamiltonian->onInteractionChanged();
    };

    virtual void disable() final
    {
        this->enabled = false;
        hamiltonian->onInteractionChanged();
    };

protected:
    ABC( Hamiltonian * hamiltonian, scalarfield energy_per_spin, scalar delta = 1e-3 ) noexcept
            : energy_per_spin( std::move( energy_per_spin ) ), delta( delta ), hamiltonian( hamiltonian ){};

    std::mt19937 prng;
    std::uniform_int_distribution<int> distribution_int;

    virtual void updateFromGeometry( const Data::Geometry * geometry ) = 0;

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

    // as long as the interaction is only constructible inside the Hamiltonian,
    // it is safe to assume that the Hamiltonian pointed to always exists
    Hamiltonian * hamiltonian;

private:
    static constexpr std::string_view name          = "Interaction::ABC";
    static constexpr std::optional<int> spin_order_ = std::nullopt;
};

/*
 * Interaction Base class to inherit from (using CRTP) when implementing an Interaction
 * CRTP is used because of the clone() and Name() methods, which are identically implemented on each subclass,
 * but they each need specific knowledge of the subclass for these implementations
 */
template<class Derived>
class Base : public Interaction::ABC
{
protected:
    Base( Hamiltonian * hamiltonian, scalarfield energy_per_spin, scalar delta = 1e-3 ) noexcept
            : ABC( hamiltonian, energy_per_spin, delta )
    {
        prng             = std::mt19937( 94199188 );
        distribution_int = std::uniform_int_distribution<int>( 0, 1 );
    };

public:
    [[nodiscard]] std::unique_ptr<ABC> clone( Hamiltonian * new_hamiltonian ) const override
    {
        auto copy         = std::make_unique<Derived>( static_cast<const Derived &>( *this ) );
        copy->hamiltonian = new_hamiltonian;
        return copy;
    }

    [[nodiscard]] std::string_view Name() const final
    {
        return Derived::name;
    }

    std::optional<int> spin_order() const final
    {
        return Derived::spin_order_;
    };

    void updateGeometry() final
    {
        this->updateFromGeometry( this->hamiltonian->geometry.get() );
    }
};

inline void setOwnerPtr( ABC & interaction, Hamiltonian * const hamiltonian ) noexcept
{
    interaction.hamiltonian = hamiltonian;
};

} // namespace Interaction

} // namespace Engine

#endif
