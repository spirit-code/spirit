#pragma once

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

namespace Functor
{

struct dense_hessian_wrapper
{
    // If this operation should be used in a parallelized algorithm this object this needs a mutex
    void operator()( const int i, const int j, const scalar value ) const
    {
        hessian( i, j ) += value;
    }

    constexpr explicit dense_hessian_wrapper( MatrixX & hessian ) : hessian( hessian ){};

private:
    MatrixX & hessian;
};

struct sparse_hessian_wrapper
{
    using triplet = Common::Interaction::triplet;

    // If this operation should be used in a parallelized algorithm this object needs a mutex
    void operator()( const int i, const int j, const scalar value ) const
    {
        hessian.emplace_back( i, j, value );
    }

    constexpr explicit sparse_hessian_wrapper( field<triplet> & hessian ) : hessian( hessian ){};

private:
    field<triplet> & hessian;
};

namespace NonLocal
{

template<typename InteractionType>
struct DataRef
{
    using Interaction = InteractionType;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;

    constexpr DataRef( const Data & data, Cache & cache ) noexcept : data( data ), cache( cache ){};

    const Data & data;
    Cache & cache;
};

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
    decltype( functor.data ) data   = functor.data;
    decltype( functor.cache ) cache = functor.cache;
};

template<typename DataRef>
struct Energy_Functor : public DataRef
{
    using Interaction = typename DataRef::Interaction;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;

    void operator()( const vectorfield & spins, scalarfield & energy ) const;

    using DataRef::DataRef;
};

template<typename DataRef>
struct Gradient_Functor : public DataRef
{
    using Interaction = typename DataRef::Interaction;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;

    void operator()( const vectorfield & spins, vectorfield & gradient ) const;

    using DataRef::DataRef;
};

template<typename DataRef>
struct Hessian_Functor : public DataRef
{
    using Interaction = typename DataRef::Interaction;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;

    template<typename Callable>
    void operator()( const vectorfield & spins, Callable & hessian ) const;

    using DataRef::DataRef;
};

template<typename DataRef>
struct Energy_Single_Spin_Functor : public DataRef
{
    using Interaction = typename DataRef::Interaction;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;

    scalar operator()( int ispin, const vectorfield & spins ) const;

    using DataRef::DataRef;
};

} // namespace NonLocal

namespace Local
{

template<typename InteractionType>
struct DataRef
{
    using Interaction = InteractionType;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;
    using Index       = typename Interaction::Index;

    constexpr DataRef( const Data & data, const Cache & cache ) noexcept : data( data ), cache( cache ){};

    const Data & data;
    const Cache & cache;
};

template<typename DataRef>
struct Energy_Functor : public DataRef
{
    using Interaction = typename DataRef::Interaction;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;
    using Index       = typename Interaction::Index;

    scalar operator()( const Index & index, const vectorfield & spins ) const;

    using DataRef::DataRef;
};

template<typename DataRef>
struct Gradient_Functor : public DataRef
{
    using Interaction = typename DataRef::Interaction;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;
    using Index       = typename Interaction::Index;

    Vector3 operator()( const Index & index, const vectorfield & spins ) const;

    using DataRef::DataRef;
};

template<typename DataRef>
struct Hessian_Functor : public DataRef
{
    using Interaction = typename DataRef::Interaction;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;
    using Index       = typename Interaction::Index;

    template<typename Callable>
    void operator()( const Index & index, const vectorfield & spins, Callable & hessian ) const;

    using DataRef::DataRef;
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
    decltype( functor.data ) data   = functor.data;
    decltype( functor.cache ) cache = functor.cache;
};

} // namespace Local

} // namespace Functor

} // namespace Interaction

} // namespace Spin

} // namespace Engine
