#pragma once

#include <engine/Span.hpp>
#include <engine/common/Interaction_Traits.hpp>
#include <engine/common/Interaction_Wrapper.hpp>
#include <engine/common/StateType.hpp>

#include <type_traits>

namespace Engine
{

namespace Common
{

namespace Accessor
{

template<typename T>
using Energy_Single_Spin = typename T::Energy_Single_Spin;

template<typename T>
using Energy = typename T::Energy;

template<typename T>
using Energy_Total = typename T::Energy_Total;

template<typename T>
using Gradient = typename T::Gradient;

template<typename T>
using Hessian = typename T::Hessian;

} // namespace Accessor

namespace Functor
{

template<template<class> class FunctorAccessor, typename... WrappedInteractionTypes>
auto tuple_dispatch( Backend::tuple<WrappedInteractionTypes...> & interactions )
    -> Backend::tuple<FunctorAccessor<typename WrappedInteractionTypes::Interaction>...>
{
    return Backend::apply(
        []( WrappedInteractionTypes &... interaction )
        { return Backend::make_tuple( Interaction::make_functor<FunctorAccessor>( interaction )... ); }, interactions );
};

template<typename... Functors, typename... Args>
void apply( Backend::tuple<Functors...> functors, Args &&... args )
{
    Backend::apply(
        [args = Backend::tuple<Args...>( std::forward<Args>( args )... )]( const Functors &... functor ) -> void
        { ( ..., Backend::apply( functor, args ) ); }, functors );
}

template<typename... Functors, typename ReturnType, typename... Args>
auto apply_reduce( Backend::tuple<Functors...> functors, ReturnType zero, Args &&... args ) -> ReturnType
{
    return Backend::apply(
        [zero, args = Backend::tuple<Args...>( std::forward<Args>( args )... )](
            const Functors &... functor ) -> ReturnType { return ( zero + ... + Backend::apply( functor, args ) ); },
        functors );
}

template<typename state_t, typename ReturnType, typename... Functors>
struct transform_op
{
    template<typename IndexTuple>
    SPIRIT_HOSTDEVICE auto operator()( const IndexTuple & index ) const -> ReturnType
    {
        return Backend::apply(
            [this, &index]( const Functors &... functor ) -> ReturnType
            {
                return (
                    zero + ... +
                    [this, &index]( const Functors & functor ) {
                        return functor( Backend::get<typename Functors::Interaction::Index>( index ), state );
                    }( functor ) );
            },
            functors );
    };

    constexpr transform_op( Backend::tuple<Functors...> && functors, ReturnType zero, const state_t & state ) noexcept(
        std::is_nothrow_move_constructible_v<Backend::tuple<Functors...>>
        && std::is_nothrow_copy_constructible_v<ReturnType> )
            : functors( std::move( functors ) ), state( state.data() ), zero( zero ) {};

private:
    Backend::tuple<Functors...> functors;
    typename state_traits<state_t>::const_pointer state;
    ReturnType zero;
};

template<typename state_t, typename UnaryOp, typename... Functors>
struct for_each_op
{
    template<typename IndexTuple>
    auto operator()( const IndexTuple & index ) const -> void
    {
        Backend::apply(
            [this, &index]( const Functors &... functor ) -> void
            {
                ( ...,
                  [this, &index]( const Functors & functor ) -> void {
                      functor( Backend::get<typename Functors::Interaction::Index>( index ), state, unary_op );
                  }( functor ) );
            },
            functors );
    };

    constexpr for_each_op(
        Backend::tuple<Functors...> && functors, const state_t & state,
        UnaryOp & unary_op ) noexcept( std::is_nothrow_move_constructible_v<Backend::tuple<Functors...>> )
            : functors( std::move( functors ) ), state( state ), unary_op( unary_op ) {};

private:
    Backend::tuple<Functors...> functors;
    const state_t & state;
    UnaryOp & unary_op;
};

} // namespace Functor

} // namespace Common

} // namespace Engine
