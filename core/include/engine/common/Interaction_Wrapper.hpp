#pragma once

#include <data/Geometry.hpp>
#include <engine/Vectormath_Defines.hpp>
#include <engine/common/Interaction_Traits.hpp>
#include <engine/common/interaction/Functor_Prototypes.hpp>

#include <memory>
#include <optional>

namespace Engine
{

namespace Common
{

namespace Interaction
{

template<typename InteractionType>
struct InteractionWrapper;

template<typename InteractionType>
class StandaloneFactory;

template<typename InteractionType>
struct InteractionWrapper
{
    using Interaction = InteractionType;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;

    static_assert(
        std::is_default_constructible<Cache>::value, "InteractionType::Cache has to be default constructible" );

    template<typename AdaptorType>
    friend class StandaloneFactory;

    explicit InteractionWrapper( typename InteractionType::Data && init_data ) : data( init_data ), cache(){};
    explicit InteractionWrapper( const typename InteractionType::Data & init_data ) : data( init_data ), cache(){};

    // applyGeometry
    void applyGeometry( const ::Data::Geometry & geometry, const intfield & boundary_conditions )
    {
        static_assert( !is_local<InteractionType>::value );
        Interaction::applyGeometry( geometry, boundary_conditions, data, cache );
    }

    template<typename IndexVector>
    void applyGeometry( const ::Data::Geometry & geometry, const intfield & boundary_conditions, IndexVector & indices )
    {
        static_assert( is_local<InteractionType>::value );
        Interaction::applyGeometry( geometry, boundary_conditions, data, cache, indices );
    }

    // is_contributing
    bool is_contributing() const
    {
        return Interaction::is_contributing( data, cache );
    }

private:
    template<typename T>
    struct has_valid_check
    {
    private:
        template<typename U>
        static auto test( U & p, typename U::Data & data ) -> decltype( p.valid_data( data ), std::true_type() );

        template<typename>
        static std::false_type test( ... );

    public:
        static constexpr bool value = decltype( test<T>( std::declval<T>(), std::declval<typename T::Data>() ) )::value;
    };

public:
    // set_data
    auto set_data( typename Interaction::Data && new_data ) -> std::optional<std::string>
    {
        if constexpr( has_valid_check<Interaction>::value )
            if( !Interaction::valid_data( new_data ) )
                return fmt::format( "the data passed to interaction \"{}\" is invalid", Interaction::name );

        data = std::move( new_data );
        return std::nullopt;
    };

    // Sparse_Hessian_Size_per_Cell
    std::size_t Sparse_Hessian_Size_per_Cell() const
    {
        return Interaction::Sparse_Hessian_Size_per_Cell( data, cache );
    }

    Data data;
    Cache cache = Cache();
};

template<template<typename> typename FunctorAccessor, typename InteractionType>
FunctorAccessor<InteractionType> make_functor( InteractionWrapper<InteractionType> & interaction ) noexcept(
    std::is_nothrow_constructible<
        FunctorAccessor<InteractionType>, typename InteractionType::Data, typename InteractionType::Cache>::value )
{
    return FunctorAccessor<InteractionType>( interaction.data, interaction.cache );
}

template<typename StandaloneFactoryType, typename... WrappedInteractionType, typename Iterator>
constexpr Iterator
generate_active_nonlocal( Backend::tuple<WrappedInteractionType...> & interactions, Iterator iterator )
{
    static_assert(
        std::conjunction<std::negation<is_local<typename WrappedInteractionType::Interaction>>...>::value,
        "all interaction types in tuple must be non-local" );

    return Backend::apply(
        [&iterator]( WrappedInteractionType &... elements )
        {
            ( ...,
              [&iterator]( WrappedInteractionType & interaction )
              {
                  if( interaction.is_contributing() )
                      *( iterator++ ) = StandaloneFactoryType::make_standalone( interaction );
              }( elements ) );

            return iterator;
        },
        interactions );
};

template<typename StandaloneFactoryType, typename... WrappedInteractionType, typename IndexVector, typename Iterator>
constexpr Iterator generate_active_local(
    Backend::tuple<WrappedInteractionType...> & interactions, const IndexVector & indices, Iterator iterator )
{
    static_assert(
        std::conjunction<is_local<typename WrappedInteractionType::Interaction>...>::value,
        "all interaction types in tuple must be local" );

    return Backend::apply(
        [&indices, &iterator]( WrappedInteractionType &... elements )
        {
            ( ...,
              [&indices, &iterator]( WrappedInteractionType & interaction )
              {
                  if( interaction.is_contributing() )
                      *( iterator++ ) = StandaloneFactoryType::make_standalone( interaction, indices );
              }( elements ) );

            return iterator;
        },
        interactions );
};

} // namespace Interaction

} // namespace Common

} // namespace Engine
