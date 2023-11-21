#pragma once
#ifndef SPIRIT_CORE_ENGINE_INTERACTION_HAMILTONIAN_HPP
#define SPIRIT_CORE_ENGINE_INTERACTION_HAMILTONIAN_HPP

#include <Spirit/Hamiltonian.h>
#include <Spirit/Spirit_Defines.h>
#include <data/Geometry.hpp>
#include <engine/FFT.hpp>
#include <engine/Hamiltonian_Defines.hpp>
#include <engine/Vectormath_Defines.hpp>
#include <utility/Span.hpp>

#include <memory>
#include <random>
#include <vector>

namespace Engine
{

class Hamiltonian;

// forward declaration of interactions for following friend declaration
namespace Interaction
{

using triplet = Eigen::Triplet<scalar>;

// common interaction base class
class ABC;

// CRTP class for shared implementations that need knowledge of the specific child class
template<class Derived>
class Base;

// Interaction classes
class Gaussian;
class Zeeman;
class Anisotropy;
class Cubic_Anisotropy;
class Exchange;
class DMI;
class DDI;
class Quadruplet;

void setOwnerPtr( ABC & interaction, Hamiltonian * hamiltonian ) noexcept;

} // namespace Interaction

enum class HAMILTONIAN_CLASS
{
    GENERIC    = SPIRIT_HAMILTONIAN_CLASS_GENERIC,
    GAUSSIAN   = SPIRIT_HAMILTONIAN_CLASS_GAUSSIAN,
    HEISENBERG = SPIRIT_HAMILTONIAN_CLASS_HEISENBERG,
};

static constexpr std::string_view hamiltonianClassName( HAMILTONIAN_CLASS cls )
{
    switch( cls )
    {
        case HAMILTONIAN_CLASS::GENERIC: return "Generic";
        case HAMILTONIAN_CLASS::GAUSSIAN: return "Gaussian";
        case HAMILTONIAN_CLASS::HEISENBERG: return "Heisenberg";
        default: return "Unknown";
    };
}

/*
    The Heisenberg Hamiltonian using Pairs contains all information on the interactions between spins.
    The information is presented in pair lists and parameter lists in order to easily e.g. calculate the energy of the
   system via summation. Calculations are made on a per-pair basis running over all pairs.
*/
class Hamiltonian
{
    // Marked as friend classes as an alternative to having a proper accessor for geometry & boundary_conditions.
    // This might also prove to be more flexible and makes sense semantically because interactions and Hamiltonian are
    // so intimately linked
    friend class Interaction::ABC;
    template<class Derived>
    friend class Interaction::Base;
    // Interaction classes
    friend class Interaction::Gaussian;
    friend class Interaction::Zeeman;
    friend class Interaction::Anisotropy;
    friend class Interaction::Cubic_Anisotropy;
    friend class Interaction::Exchange;
    friend class Interaction::DMI;
    friend class Interaction::DDI;
    friend class Interaction::Quadruplet;

public:
    friend void swap( Hamiltonian & first, Hamiltonian & second ) noexcept
    {
        using std::swap;
        if( &first == &second )
        {
            return;
        }
        swap( first.geometry, second.geometry );
        swap( first.boundary_conditions, second.boundary_conditions );
        swap( first.name_update_paused, second.name_update_paused );
        swap( first.hamiltonian_class, second.hamiltonian_class );
        swap( first.class_name, second.class_name );

        swap( first.interactions, second.interactions );
        swap( first.active_interactions_size, second.active_interactions_size );
        swap( first.common_interactions_size, second.common_interactions_size );

        swap( first.prng, second.prng );
        swap( first.distribution_int, second.distribution_int );
        swap( first.delta, second.delta );

        for( const auto & interaction : first.interactions )
            Interaction::setOwnerPtr( *interaction, &first );

        for( const auto & interaction : second.interactions )
            Interaction::setOwnerPtr( *interaction, &second );
    }

    Hamiltonian( std::shared_ptr<Data::Geometry> geometry, intfield boundary_conditions );

    Hamiltonian() = default;
    // rule of five, because the Hamiltonian owns the interactions
    Hamiltonian( const Hamiltonian & other );
    // using the copy-and-swap idiom for brevity. Should be implemented more efficiently once the refactor is finished.
    Hamiltonian & operator=( Hamiltonian other )
    {
        swap( *this, other );
        return *this;
    };

    Hamiltonian( Hamiltonian && other ) noexcept : Hamiltonian()
    {
        swap( *this, other );
    };

    Hamiltonian & operator=( Hamiltonian && other ) noexcept
    {
        swap( *this, other );
        return *this;
    };

    ~Hamiltonian() = default;

    Utility::Span<const std::unique_ptr<Interaction::ABC>> getActiveInteractions() const
    {
        return Utility::Span( interactions.begin(), active_interactions_size );
    }

    std::size_t getActiveInteractionsSize() const
    {
        return active_interactions_size;
    };

    Utility::Span<const std::unique_ptr<Interaction::ABC>> getInactiveInteractions() const
    {
        return Utility::Span( interactions.begin() + active_interactions_size, interactions.end() );
    }

    std::size_t getInactiveInteractionsSize() const
    {
        return interactions.size() - active_interactions_size;
    };

    const auto & getInteractions() const
    {
        return interactions;
    };

    std::size_t getInteractionsSize() const
    {
        return interactions.size();
    };

    /*
     * Update the internal state of the interactions.
     * This needs to be done every time the parameters are changed, in case an energy
     * contribution is now non-zero or vice versa.
     * The interactions know when to trigger this if the parameters are updated for them directly,
     * the manual update should only be neccessary if the geometry or boundary_conditions are changed.
     */
    void updateInteractions();
    void updateActiveInteractions();

    /*
     * update functions for when the geometry or an Interaction has been changed,
     * these serve as an alternative to more rigid accessors
     */
    void onInteractionChanged()
    {
        updateActiveInteractions();
    };
    void onGeometryChanged()
    {
        updateInteractions();
    };

    void Hessian( const vectorfield & spins, MatrixX & hessian );
    void Sparse_Hessian( const vectorfield & spins, SpMatrixX & hessian );

    void Gradient( const vectorfield & spins, vectorfield & gradient );
    void Gradient_and_Energy( const vectorfield & spins, vectorfield & gradient, scalar & energy );

    void Energy_Contributions_per_Spin( const vectorfield & spins, Data::vectorlabeled<scalarfield> & contributions );
    Data::vectorlabeled<scalar> Energy_Contributions( const vectorfield & spins );

    // Calculate the total energy for a single spin to be used in Monte Carlo.
    //      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
    scalar Energy_Single_Spin( int ispin, const vectorfield & spins );

    scalar Energy( const vectorfield & spins );

    void Gradient_FD( const vectorfield & spins, vectorfield & gradient );
    void Hessian_FD( const vectorfield & spins, MatrixX & hessian );

    // Hamiltonian name as string
    void pauseUpdateName()
    {
        name_update_paused = true;
    };

    void unpauseUpdateName()
    {
        name_update_paused = false;
    };

    void updateName();
    std::string_view Name() const;

    Interaction::ABC * getInteraction( std::string_view name );
    std::size_t deleteInteraction( std::string_view name );

    template<class T, typename... Args>
    T & setInteraction( Args &&... args );

    template<class T>
    const T * getInteraction() const;

    template<class T>
    T * getInteraction();

    template<class T>
    bool hasInteraction();

    template<class T>
    std::size_t deleteInteraction();

    std::shared_ptr<Data::Geometry> geometry;
    intfield boundary_conditions;

private:
    // common and uncommon interactions partition the active interactions
    Utility::Span<const std::unique_ptr<Interaction::ABC>> getCommonInteractions() const
    {
        return Utility::Span( interactions.begin(), common_interactions_size );
    };

    Utility::Span<const std::unique_ptr<Interaction::ABC>> getUncommonInteractions() const
    {
        return Utility::Span(
            interactions.begin() + common_interactions_size, active_interactions_size - common_interactions_size );
    };

    std::vector<std::unique_ptr<Interaction::ABC>> interactions{};
    std::size_t active_interactions_size = 0;
    std::size_t common_interactions_size = 0;

    std::mt19937 prng;
    std::uniform_int_distribution<int> distribution_int;
    scalar delta = 1e-3;

    // naming mechanism for compatibility with the named subclasses architecture
    bool name_update_paused             = false;
    HAMILTONIAN_CLASS hamiltonian_class = HAMILTONIAN_CLASS::GENERIC;
    std::string_view class_name{ hamiltonianClassName( HAMILTONIAN_CLASS::GENERIC ) };

    static constexpr int common_spin_order = 2;
};

} // namespace Engine

#endif
