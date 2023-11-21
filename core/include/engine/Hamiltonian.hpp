#pragma once
#ifndef SPIRIT_CORE_ENGINE_HAMILTONIAN_HPP
#define SPIRIT_CORE_ENGINE_HAMILTONIAN_HPP

#include <engine/interaction/ABC.hpp>
#include <engine/interaction/Anisotropy.hpp>
#include <engine/interaction/Cubic_Anisotropy.hpp>
#include <engine/interaction/DDI.hpp>
#include <engine/interaction/DMI.hpp>
#include <engine/interaction/Exchange.hpp>
#include <engine/interaction/Gaussian.hpp>
#include <engine/interaction/Hamiltonian.hpp>
#include <engine/interaction/Quadruplet.hpp>
#include <engine/interaction/Zeeman.hpp>

namespace Engine
{

// runtime getter
inline Interaction::ABC * Hamiltonian::getInteraction( std::string_view name )
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
inline std::size_t Hamiltonian::deleteInteraction( std::string_view name )
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
T & Hamiltonian::setInteraction( Args &&... args )
{
    static_assert( std::is_convertible_v<T *, Interaction::Base<T> *>, "T has to be derived from Interaction::Base" );
    static_assert( std::is_constructible_v<T, Engine::Hamiltonian *, Args...>, "No matching constructor for T" );

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
const T * Hamiltonian::getInteraction() const
{
    static_assert( std::is_convertible_v<T *, Interaction::Base<T> *>, "T has to be derived from Interaction::Base" );
    return dynamic_cast<T *>( getInteraction( T::name ) );
};

// compile time getter
template<class T>
T * Hamiltonian::getInteraction()
{
    static_assert( std::is_convertible_v<T *, Interaction::Base<T> *>, "T has to be derived from Interaction::Base" );
    return dynamic_cast<T *>( getInteraction( T::name ) );
};

template<class T>
bool Hamiltonian::hasInteraction()
{
    return ( getInteraction<T>() != nullptr );
};

// compile time deleter
template<class T>
std::size_t Hamiltonian::deleteInteraction()
{
    static_assert( std::is_convertible_v<T *, Interaction::Base<T> *>, "T has to be derived from Interaction::Base" );
    return deleteInteraction( T::name );
};

} // namespace Engine
#endif
