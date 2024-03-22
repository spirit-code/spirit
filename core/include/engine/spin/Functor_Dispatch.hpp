#pragma once

#include <engine/spin/Interaction_Traits.hpp>
#include <engine/spin/Interaction_Wrapper.hpp>

#include <tuple>
#include <type_traits>

namespace Engine
{

namespace Spin
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

template<template<class> class FunctorAccessor, typename... InteractionTypes>
auto tuple_dispatch( std::tuple<Interaction::InteractionWrapper<InteractionTypes>...> & interactions )
    -> std::tuple<FunctorAccessor<InteractionTypes>...>
{
    return std::apply(
        []( Interaction::InteractionWrapper<InteractionTypes> &... interaction )
        { return std::make_tuple( interaction.template make_functor<FunctorAccessor>()... ); },
        interactions );
};

template<typename... Functors, typename... Args>
void apply( std::tuple<Functors...> functors, Args &&... args )
{
    std::apply(
        [args = std::tuple<Args...>( std::forward<Args>( args )... )]( const Functors &... functor ) -> void
        { ( ..., std::apply( functor, args ) ); },
        functors );
}

template<typename... Functors, typename ReturnType, typename... Args>
auto apply_reduce( std::tuple<Functors...> functors, ReturnType zero, Args &&... args ) -> ReturnType
{
    return std::apply(
        [zero, args = std::tuple<Args...>( std::forward<Args>( args )... )]( const Functors &... functor ) -> ReturnType
        { return ( zero + ... + std::apply( functor, args ) ); },
        functors );
}

template<typename state_t, typename ReturnType, typename... Functors>
struct transform_op
{
    template<typename IndexTuple>
    SPIRIT_HOSTDEVICE auto operator()( const IndexTuple & index ) const -> ReturnType
    {
        return std::apply(
            [this, &index]( const Functors &... functor ) -> ReturnType
            {
                return (
                    zero + ... +
                    [this, &index]( const Functors & functor )
                    { return functor( std::get<typename Functors::Interaction::Index>( index ), state ); }( functor ) );
            },
            functors );
    };

    constexpr transform_op( std::tuple<Functors...> && functors, ReturnType zero, const state_t & state ) noexcept(
        std::is_nothrow_move_constructible_v<std::tuple<Functors...>>
        && std::is_nothrow_copy_constructible_v<ReturnType> )
            : functors( functors ), state( raw_pointer_cast( state.data() ) ), zero( zero ){};

private:
    std::tuple<Functors...> functors;
    const std::decay_t<typename state_t::value_type> * state;
    ReturnType zero;
};

template<typename state_t, typename UnaryOp, typename... Functors>
struct for_each_op
{
    template<typename IndexTuple>
    auto operator()( const IndexTuple & index ) const -> void
    {
        std::apply(
            [this, &index]( const Functors &... functor ) -> void
            {
                ( ...,
                  [this, &index]( const Functors & functor ) -> void {
                      functor( std::get<typename Functors::Interaction::Index>( index ), state, unary_op );
                  }( functor ) );
            },
            functors );
    };

    constexpr for_each_op( std::tuple<Functors...> && functors, const state_t & state, UnaryOp & unary_op ) noexcept(
        std::is_nothrow_move_constructible_v<std::tuple<Functors...>> )
            : functors( functors ), state( state ), unary_op( unary_op ){};

private:
    std::tuple<Functors...> functors;
    const state_t & state;
    UnaryOp & unary_op;
};

} // namespace Functor

} // namespace Spin

} // namespace Engine
