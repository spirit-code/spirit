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

#include <numeric>
#include <vector>

namespace Engine
{

namespace Spin
{

namespace Interaction
{

using Common::Interaction::triplet;

struct dense_hessian_functor
{
    void operator()( const int i, const int j, const scalar value ) const
    {
        hessian( i, j ) += value;
    }

    constexpr explicit dense_hessian_functor( MatrixX & hessian ) : hessian( hessian ){};

private:
    MatrixX & hessian;
};

struct sparse_hessian_functor
{
    void operator()( const int i, const int j, const scalar value ) const
    {
        hessian.emplace_back( i, j, value );
    }

    constexpr explicit sparse_hessian_functor( field<triplet> & hessian ) : hessian( hessian ){};

private:
    field<triplet> & hessian;
};

namespace NonLocal
{

template<typename Functor>
struct Reduce_Functor
{
    using Interaction = typename Functor::Interaction;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;

    template<typename... Args>
    constexpr Reduce_Functor( Args &&... args ) noexcept( std::is_nothrow_constructible_v<Functor, Args...> )
            : functor( std::forward<Args>( args )... ){};

    scalar operator()( const typename Interaction::state_t & state ) const
    {
        scalarfield energy_per_spin( state.size() );
        functor( state, energy_per_spin );
        return std::reduce( begin( energy_per_spin ), end( energy_per_spin ) );
    };

private:
    Functor functor;

public:
    const Data & data = functor.data;
    Cache & cache     = functor.cache;
};

template<typename InteractionType>
struct Energy_Functor
{
    using Interaction = InteractionType;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;

    void operator()( const vectorfield & spins, scalarfield & energy ) const;

    constexpr Energy_Functor( const Data & data, Cache & cache ) noexcept : data( data ), cache( cache ){};

    const Data & data;
    Cache & cache;
};

template<typename InteractionType>
struct Gradient_Functor
{
    using Interaction = InteractionType;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;

    void operator()( const vectorfield & spins, vectorfield & gradient ) const;

    constexpr Gradient_Functor( const Data & data, Cache & cache ) noexcept : data( data ), cache( cache ){};

    const Data & data;
    Cache & cache;
};

template<typename InteractionType>
struct Hessian_Functor
{
    using Interaction = InteractionType;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;
    template<typename F>
    void operator()( const vectorfield & spins, F & f ) const;

    constexpr Hessian_Functor( const Data & data, Cache & cache ) noexcept : data( data ), cache( cache ){};

    const Data & data;
    Cache & cache;
};

template<typename InteractionType>
struct Energy_Single_Spin_Functor
{
    using Interaction = InteractionType;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;

    scalar operator()( int ispin, const vectorfield & spins ) const;

    constexpr Energy_Single_Spin_Functor( const Data & data, Cache & cache ) noexcept : data( data ), cache( cache ){};

    const Data & data;
    Cache & cache;
};

// template<typename GradientFunctor, typename TotalEnergyFunctor>
// struct Sequential_Gradient_and_Energy_Functor
// {
//     static_assert( std::is_same_v<typename GradientFunctor::Interaction, typename TotalEnergyFunctor::Interaction> );
//     using Interaction = typename GradientFunctor::Interaction;
//     using Data        = typename Interaction::Data;
//     using Cache       = typename Interaction::Cache;
//
//     constexpr Sequential_Gradient_and_Energy_Functor( const Data & data, Cache & cache ) noexcept(
//         std::is_nothrow_constructible_v<GradientFunctor, const Data &, Cache &>
//         && std::is_nothrow_constructible_v<TotalEnergyFunctor, const Data &, Cache &> )
//             : gradient_func( data, cache ), energy_func( data, cache ){};
//
//     scalar operator()( const typename Interaction::state_t & state, vectorfield & gradient ) const
//     {
//         gradient_func( index, state, gradient );
//         return energy_func( index, state );
//     };
//
// private:
//     GradientFunctor gradient_func;
//     TotalEnergyFunctor energy_func;
// };

} // namespace NonLocal

namespace Local
{

template<typename InteractionType>
struct Energy_Functor
{
    using Interaction = InteractionType;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;
    using Index       = typename Interaction::Index;

    scalar operator()( const Index & index, const vectorfield & spins ) const;

    constexpr Energy_Functor( const Data & data, const Cache & cache ) noexcept : data( data ), cache( cache ){};

    const Data & data;
    const Cache & cache;
};

template<typename InteractionType>
struct Gradient_Functor
{
    using Interaction = InteractionType;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;
    using Index       = typename Interaction::Index;

    Vector3 operator()( const Index & index, const vectorfield & spins ) const;

    constexpr Gradient_Functor( const Data & data, const Cache & cache ) noexcept : data( data ), cache( cache ){};

    const Data & data;
    const Cache & cache;
};

template<typename InteractionType>
struct Hessian_Functor
{
    using Interaction = InteractionType;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;
    using Index       = typename Interaction::Index;

    template<typename F>
    void operator()( const Index & index, const vectorfield & spins, F & f ) const;

    constexpr Hessian_Functor( const Data & data, const Cache & cache ) noexcept : data( data ), cache( cache ){};

    const Data & data;
    const Cache & cache;
};

template<typename Functor, int weight_factor>
struct Energy_Single_Spin_Functor
{
    using Interaction = typename Functor::Interaction;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;
    using Index       = typename Interaction::Index;

    template<typename... Args>
    constexpr Energy_Single_Spin_Functor( Args &&... args ) noexcept(
        std::is_nothrow_constructible_v<Functor, Args...> )
            : functor( std::forward<Args>( args )... ){};

    scalar operator()( const Index & index, const typename Interaction::state_t & state ) const
    {
        return weight * functor( index, state );
    };

private:
    Functor functor;
    static constexpr scalar weight = weight_factor;

public:
    const Data & data   = functor.data;
    const Cache & cache = functor.cache;
};

// template<typename GradientFunctor, typename EnergyFunctor>
// struct Sequential_Gradient_and_Energy_Functor
// {
//     static_assert( std::is_same_v<typename GradientFunctor::Interaction, typename EnergyFunctor::Interaction> );
//     using Interaction = typename GradientFunctor::Interaction;
//     using Data        = typename Interaction::Data;
//     using Cache       = typename Interaction::Cache;
//     using Index       = typename Interaction::Index;
//
//     constexpr Sequential_Gradient_and_Energy_Functor( const Data & data, Cache & cache ) noexcept(
//         std::is_nothrow_constructible_v<GradientFunctor, const Data &, Cache &>
//         && std::is_nothrow_constructible_v<EnergyFunctor, const Data &, Cache &> )
//             : gradient_func( data, cache ), energy_func( data, cache ){};
//
//     void operator()(
//         const Index & index, const typename Interaction::state_t & state, Vector3 & gradient, scalar & energy ) const
//     {
//         gradient = gradient_func( index, state );
//         energy   = energy_func( index, state );
//     };
//
// private:
//     GradientFunctor gradient_func;
//     EnergyFunctor energy_func;
// };
//
// template<typename GradientFunctor, std::size_t spin_order>
// struct Monomial_Gradient_and_Energy_Functor
// {
//     static_assert( spin_order > 0 );
//
//     using Interaction = typename GradientFunctor::Interaction;
//     using Data        = typename Interaction::Data;
//     using Cache       = typename Interaction::Cache;
//     using Index       = typename Interaction::Index;
//
//     template<typename... Args>
//     constexpr Monomial_Gradient_and_Energy_Functor( Args &&... args ) noexcept(
//         std::is_nothrow_constructible_v<GradientFunctor, Args...> )
//             : gradient_func( std::forward<Args>( args )... ){};
//
//     std::pair<Vector3, scalar> operator()( const Index & index, const typename Interaction::state_t & state ) const
//     {
//         const Vector3 gradient = gradient_func( index, state );
//         const scalar energy    = prefactor * gradient.dot( state[index.ispin] );
//         return { gradient, energy };
//     };
//
// private:
//     GradientFunctor gradient_func;
//     static constexpr scalar prefactor = 1.0 / scalar( spin_order );
// };

} // namespace Local

} // namespace Interaction

} // namespace Spin

} // namespace Engine

#endif
