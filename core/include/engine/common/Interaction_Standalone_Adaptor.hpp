#pragma once

#include <engine/common/Interaction_Wrapper.hpp>

namespace Engine
{

namespace Common
{

namespace Interaction
{

template<typename state_type>
struct StandaloneAdaptor
{
    virtual ~StandaloneAdaptor()                               = default;
    StandaloneAdaptor( const StandaloneAdaptor & )             = delete;
    StandaloneAdaptor( StandaloneAdaptor && )                  = delete;
    StandaloneAdaptor & operator=( const StandaloneAdaptor & ) = delete;
    StandaloneAdaptor & operator=( StandaloneAdaptor && )      = delete;

    using state_t = state_type;

    virtual scalar Energy( const state_t & state )                                       = 0;
    virtual void Energy_per_Spin( const state_t & state, scalarfield & energy_per_spin ) = 0;
    virtual scalar Energy_Single_Spin( int ispin, const state_t & state )                = 0;
    virtual std::string_view Name() const                                                = 0;

protected:
    constexpr StandaloneAdaptor() = default;
};

template<typename InteractionType, typename AdaptorInterface>
class StandaloneAdaptor_NonLocal : public AdaptorInterface
{
    static_assert(
        !is_local<InteractionType>::value, "interaction type for non-local standalone adaptor must be non-local" );

    using Interaction = InteractionType;
    using Data        = typename InteractionType::Data;
    using Cache       = typename InteractionType::Cache;

public:
    using state_t = typename InteractionType::state_t;

    scalar Energy( const state_t & state ) final
    {
        return std::invoke( typename InteractionType::Energy_Total( data, cache ), state );
    }
    void Energy_per_Spin( const state_t & state, scalarfield & energy_per_spin ) final
    {
        std::invoke( typename InteractionType::Energy( data, cache ), state, energy_per_spin );
    }

    scalar Energy_Single_Spin( const int ispin, const state_t & state ) final
    {
        return std::invoke( typename InteractionType::Energy_Single_Spin( data, cache ), ispin, state );
    }

    std::string_view Name() const final
    {
        return InteractionType::name;
    }

protected:
    StandaloneAdaptor_NonLocal( const Data & data, Cache & cache ) noexcept
            : AdaptorInterface(), data( data ), cache( cache ){};

    const Data & data;
    Cache & cache;
};

template<typename InteractionType, typename AdaptorInterface, typename IndexVector>
class StandaloneAdaptor_Local : public AdaptorInterface
{
    static_assert( is_local<InteractionType>::value, "interaction type for local standalone adaptor must be local" );

    using Interaction = InteractionType;
    using Data        = typename InteractionType::Data;
    using Cache       = typename InteractionType::Cache;
    using IndexTuple  = typename IndexVector::value_type;

public:
    using state_t = typename InteractionType::state_t;

    scalar Energy( const state_t & state ) final
    {
        using std::begin, std::end;
        auto functor           = typename InteractionType::Energy( data, cache );
        const auto * state_ptr = raw_pointer_cast( state.data() );
        return Backend::transform_reduce(
            SPIRIT_PAR indices.begin(), indices.end(), scalar( 0.0 ), Backend::plus<scalar>{},
            [state_ptr, functor] SPIRIT_LAMBDA( const IndexTuple & index )
            { return functor( Backend::get<typename InteractionType::Index>( index ), state_ptr ); } );
    }

    void Energy_per_Spin( const state_t & state, scalarfield & energy_per_spin ) final
    {
        using std::begin, std::end;
        auto functor           = typename InteractionType::Energy( data, cache );
        const auto * state_ptr = raw_pointer_cast( state.data() );
        Backend::transform(
            SPIRIT_PAR indices.begin(), indices.end(), energy_per_spin.begin(),
            [state_ptr, functor] SPIRIT_LAMBDA( const IndexTuple & index )
            { return functor( Backend::get<typename InteractionType::Index>( index ), state_ptr ); } );
    }

    scalar Energy_Single_Spin( const int ispin, const state_t & state ) final
    {
        return std::invoke(
            typename InteractionType::Energy( data, cache ),
            Backend::get<typename InteractionType::Index>( indices[ispin] ), raw_pointer_cast( state.data() ) );
    }

    std::string_view Name() const final
    {
        return InteractionType::name;
    }

protected:
    StandaloneAdaptor_Local( const Data & data, Cache & cache, const IndexVector & indices ) noexcept
            : AdaptorInterface(), data( data ), cache( cache ), indices( indices ){};

    // all member variables must be read only to avoid race conditions in the parallelized Backend
    const Data & data;
    const Cache & cache;
    const IndexVector & indices;
};

} // namespace Interaction

} // namespace Common

} // namespace Engine
