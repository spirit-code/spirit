#pragma once

#include <Spirit/Hamiltonian.h>
#include <Spirit/Spirit_Defines.h>
#include <data/Geometry.hpp>
#include <engine/Span.hpp>
#include <engine/Backend.hpp>
#include <engine/Vectormath_Defines.hpp>

#include <vector>

namespace Engine
{

namespace Common
{

namespace Interaction
{

using triplet = Eigen::Triplet<scalar>;

template<typename IndexType>
Engine::Span<const IndexType> make_index( const Backend::vector<IndexType> & index_storage )
{
    using std::begin, std::end;
    return Engine::Span<const IndexType>( index_storage.begin(), index_storage.end() );
}

template<typename IndexType>
const IndexType * make_index( const Backend::optional<IndexType> & index_storage )
{
    if( index_storage.has_value() )
        return &( *index_storage );
    else
        return nullptr;
}

namespace Functor
{


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
        using std::begin, std::end;
        scalarfield energy_per_spin( state.size() );
        functor( state, energy_per_spin );
        return Backend::cpu::reduce( SPIRIT_CPU_PAR begin( energy_per_spin ), end( energy_per_spin ) );
    };

private:
    Functor functor;

public:
    bool is_contributing = functor.is_contributing;
};

} // namespace NonLocal

namespace Local
{

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

    template<typename... OpArgs>
    SPIRIT_HOSTDEVICE scalar operator()( OpArgs && ... args ) const
    {
        return weight * functor( std::forward<OpArgs>(args)... );
    };

private:
    Functor functor;
    static constexpr scalar weight = weight_factor;

public:
    const bool is_contributing = functor.is_contributing;
};

} // namespace Local

} // namespace Functor

} // namespace Interaction

} // namespace Spin

} // namespace Engine
