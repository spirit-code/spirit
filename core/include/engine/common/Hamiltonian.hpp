#pragma once

#include <Spirit/Hamiltonian.h>
#include <Spirit/Spirit_Defines.h>
#include <cstddef>
#include <data/Geometry.hpp>
#include <engine/FFT.hpp>
#include <engine/Vectormath_Defines.hpp>
#include <engine/common/interaction/ABC.hpp>
#include <utility/Span.hpp>

#include <memory>

namespace Engine
{

namespace Common
{

/*
 * Generic Base class for a Hamiltonian owning a collection of Interactions
 * This base class provides:
 *  - the rule of five constructor set to handle a std::vector<std::unique_ptr> member variable
 *  - an implementation for partitioning the interactions into active and inactive
 *  - getters for spans of all interactions, the active partition, and the inactive partition of interactions
 *  - implementations for the Owner interface
 *  - accessors for individual interactions (getter, setter and deleter)
 *  - a configurable naming scheme for the collection of interactions
 */
template<typename interaction_t>
class Hamiltonian : public Interaction::Owner
{
public:
    using value_t              = std::unique_ptr<interaction_t>;
    using container_t          = std::vector<value_t>;
    using value_iterator       = typename container_t::iterator;
    using value_const_iterator = typename container_t::const_iterator;

    friend void swap( Hamiltonian & first, Hamiltonian & second ) noexcept
    {
        using std::swap;
        if( &first == &second )
            return;

        swap( static_cast<Interaction::Owner &>( first ), static_cast<Interaction::Owner &>( second ) );
        swap( first.active_interactions_size, second.active_interactions_size );
        swap( first.name_update_paused, second.name_update_paused );
        // interactions
        swap( first.interactions, second.interactions );
        first.setAsOwner();
        second.setAsOwner();
    };

    // rule of five, because the Container owns the interactions
    Hamiltonian() noexcept : Interaction::Owner( nullptr, {} ){};
    Hamiltonian( const Hamiltonian & other )
            : Interaction::Owner( static_cast<const Interaction::Owner &>( other ) ),
              interactions( 0 ),
              active_interactions_size( other.active_interactions_size ),
              name_update_paused( other.name_update_paused )
    {
        // interactions
        interactions.reserve( other.interactions.size() );
        std::transform(
            begin( other.interactions ), end( other.interactions ), std::back_inserter( interactions ),
            [this]( const auto & interaction ) { return interaction->clone( this ); } );
    };

    // using the copy-and-swap idiom for brevity. Should be implemented more efficiently once the refactor is finished.
    Hamiltonian & operator=( const Hamiltonian & other )
    {
        if( this != &other )
        {
            static_cast<Interaction::Owner &>( *this ) = static_cast<const Interaction::Owner &>( other );
            active_interactions_size                   = other.active_interactions_size;
            name_update_paused                         = other.name_update_paused;

            // interactions
            interactions.clear();
            interactions.reserve( other.interactions.size() );
            std::transform(
                begin( other.interactions ), end( other.interactions ), std::back_inserter( interactions ),
                [this]( const auto & interaction ) { return interaction->clone( this ); } );
        }
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

    void onGeometryChanged() final
    {
        std::for_each(
            begin( interactions ), end( interactions ), []( auto & interaction ) { interaction->updateGeometry(); } );
        onInteractionChanged();
    };

    void onBoundaryConditionsChanged() final
    {
        std::for_each(
            begin( interactions ), end( interactions ), []( auto & interaction ) { interaction->updateGeometry(); } );
        onInteractionChanged();
    };

    /*
     * Update the internal state of the interactions.
     * This needs to be done every time the parameters are changed, in case an energy
     * contribution is now non-zero or vice versa.
     * The interactions know when to trigger this if the parameters are updated for them directly,
     * the manual update should only be neccessary if the geometry or boundary_conditions are changed.
     */
    void onInteractionChanged() final
    {
        // take inventory and put the interactions that contribute to the front of the vector
        const auto is_active                 = []( const auto & i ) { return i->is_active(); };
        const auto active_partition_boundary = std::partition( begin( interactions ), end( interactions ), is_active );
        active_interactions_size             = std::distance( begin( interactions ), active_partition_boundary );

        partitionActiveInteractions( begin( interactions ), active_partition_boundary );
    }
    // customization point for further refinement of the active interactions
    virtual void partitionActiveInteractions( value_iterator, value_iterator ){};

    void updateActiveInteractions()
    {
        onInteractionChanged();
    };

    [[nodiscard]] constexpr auto getInteractions() noexcept -> Utility::Span<value_t>
    {
        return Utility::Span( begin( interactions ), end( interactions ) );
    }

    [[nodiscard]] constexpr auto getInteractions() const noexcept -> Utility::Span<const value_t>
    {
        return Utility::Span( begin( interactions ), end( interactions ) );
    }

    [[nodiscard]] constexpr auto getActiveInteractions() const noexcept -> Utility::Span<const value_t>
    {
        return Utility::Span( begin( interactions ), active_interactions_size );
    }

    [[nodiscard]] constexpr auto getInactiveInteractions() const noexcept
    {
        return Utility::Span( begin( interactions ) + active_interactions_size, end( interactions ) );
    }

    [[nodiscard]] constexpr auto getActiveInteractionsSize() const noexcept
    {
        return active_interactions_size;
    }

    // runtime getter
    [[nodiscard]] interaction_t * getInteraction( std::string_view name )
    {
        // linear search, because setting and getting an interaction doesn't have to be fast, but iterating over all
        // (active) interactions has to be.
        auto same_name = [&name]( const auto & i ) { return name == i->Name(); };
        auto itr       = std::find_if( begin( interactions ), end( interactions ), same_name );
        if( itr != end( interactions ) )
            return itr->get();
        return nullptr;
    };

    // runtime deleter
    std::size_t deleteInteraction( std::string_view name )
    {
        auto same_name = [&name]( const auto & i ) { return name == i->Name(); };
        auto itr       = std::find_if( begin( interactions ), end( interactions ), same_name );
        if( itr != end( interactions ) )
        {
            interactions.erase( itr );
            return 1;
        }
        return 0;
    }

    // compile time setter (has to be wrapped appropriately to be used at runtime)
    template<class T, typename... Args>
    T & setInteraction( Args &&... args )
    {
        // static_assert( std::is_convertible_v<T *, interaction_t *>, "T has to be derived from Interaction::Base" );
        static_assert( std::is_constructible_v<T, Interaction::Owner *, Args...>, "No matching constructor for T" );

        auto same_name = []( const auto & i ) { return T::name == i->Name(); };
        auto itr       = std::find_if( begin( interactions ), end( interactions ), same_name );
        if( itr != end( interactions ) )
        {
            **itr = T( this, std::forward<Args>( args )... );
            this->updateName();
            return static_cast<T &>( **itr );
        }

        interactions.emplace_back( std::make_unique<T>( this, std::forward<Args>( args )... ) );
        this->updateName();
        return static_cast<T &>( *interactions.back() );
    };

    // compile time getter
    template<class T>
    [[nodiscard]] const T * getInteraction() const
    {
        static_assert( std::is_convertible_v<T *, interaction_t *>, "T has to be derived from Interaction::Base" );
        return dynamic_cast<T *>( getInteraction( T::name ) );
    };

    // compile time getter
    template<class T>
    [[nodiscard]] T * getInteraction()
    {
        static_assert( std::is_convertible_v<T *, interaction_t *>, "T has to be derived from Interaction::Base" );
        return dynamic_cast<T *>( getInteraction( T::name ) );
    };

    template<class T>
    [[nodiscard]] bool hasInteraction()
    {
        return ( getInteraction<T>() != nullptr );
    };

    // compile time deleter
    template<class T>
    std::size_t deleteInteraction()
    {
        static_assert( std::is_convertible_v<T *, interaction_t *>, "T has to be derived from Interaction::Base" );
        return deleteInteraction( T::name );
    };

    // naming mechanism
    void updateName()
    {
        if( name_update_paused )
            return;
        updateName_Impl();
    };
    virtual void updateName_Impl(){};
    [[nodiscard]] virtual std::string_view Name() const
    {
        return "Unknown";
    };

    // Hamiltonian name as string
    void pauseUpdateName()
    {
        name_update_paused = true;
    };

    void unpauseUpdateName()
    {
        name_update_paused = false;
    };

protected:
    Hamiltonian( std::shared_ptr<Data::Geometry> && geometry, intfield && boundary_conditions ) noexcept
            : Owner( geometry, boundary_conditions ){};
    Hamiltonian( const std::shared_ptr<Data::Geometry> & geometry, const intfield & boundary_conditions ) noexcept
            : Owner( geometry, boundary_conditions ){};

    container_t interactions{};

private:
    std::size_t active_interactions_size = 0;
    bool name_update_paused              = false;

    void setAsOwner()
    {
        std::for_each(
            begin( interactions ), end( interactions ), [this]( auto & i ) { Interaction::setOwnerPtr( *i, this ); } );
    }
};

} // namespace Common

} // namespace Engine
