#pragma once

#include <Spirit/Hamiltonian.h>
#include <Spirit/Spirit_Defines.h>
#include <data/Geometry.hpp>
#include <engine/Backend.hpp>
#include <engine/Span.hpp>
#include <engine/Vectormath_Defines.hpp>
#include <engine/common/interaction/Functor_Prototypes.hpp>

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

    constexpr explicit dense_hessian_wrapper( MatrixX & hessian ) : hessian( hessian ) {};

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

    constexpr explicit sparse_hessian_wrapper( std::vector<triplet> & hessian ) : hessian( hessian ) {};

private:
    std::vector<triplet> & hessian;
};

namespace NonLocal
{

using Common::Interaction::Functor::NonLocal::Reduce_Functor;

template<typename InteractionType>
struct DataRef
{
    using Interaction = InteractionType;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;

    constexpr DataRef( const Data & data, Cache & cache ) noexcept : data( data ), cache( cache ) {};

    const Data & data;
    Cache & cache;
    bool is_contributing = Interaction::is_contributing( data, cache );
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
struct Energy_Single_Spin_Functor : public DataRef
{
    using Interaction = typename DataRef::Interaction;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;

    scalar operator()( int ispin, const vectorfield & spins ) const;

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

} // namespace NonLocal

namespace Local
{

using Common::Interaction::Functor::Local::Energy_Single_Spin_Functor;

template<typename InteractionType>
struct DataRef
{
    using Interaction = InteractionType;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;
    using Index       = typename Interaction::Index;

    constexpr DataRef( const Data & data, const Cache & cache ) noexcept : data( data ), cache( cache ) {};

    const Data & data;
    const Cache & cache;
    const bool is_contributing = Interaction::is_contributing( data, cache );
};

template<typename DataRef>
struct Energy_Functor : public DataRef
{
    using Interaction = typename DataRef::Interaction;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;
    using Index       = typename Interaction::Index;

    SPIRIT_HOSTDEVICE scalar operator()( const Index & index, const Vector3 * spins ) const;

    using DataRef::DataRef;
};

template<typename DataRef>
struct Gradient_Functor : public DataRef
{
    using Interaction = typename DataRef::Interaction;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;
    using Index       = typename Interaction::Index;

    SPIRIT_HOSTDEVICE Vector3 operator()( const Index & index, const Vector3 * spins ) const;

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

} // namespace Local

} // namespace Functor

} // namespace Interaction

} // namespace Spin

} // namespace Engine
