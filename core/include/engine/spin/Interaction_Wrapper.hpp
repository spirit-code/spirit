#pragma once

#include <data/Geometry.hpp>
#include <engine/Vectormath_Defines.hpp>
#include <engine/spin/Interaction_Traits.hpp>
#include <engine/spin/interaction/Functor_Prototpyes.hpp>
#include <utility/Exception.hpp>

#include <memory>
#include <numeric>
#include <optional>
#include <string_view>
#include <tuple>

namespace Engine
{

namespace Spin
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
    virtual void Gradient( const state_t & state, vectorfield & gradient )               = 0;
    virtual void Hessian( const state_t & state, MatrixX & hessian )                     = 0;
    virtual std::string_view Name() const                                                = 0;

protected:
    constexpr explicit StandaloneAdaptor() noexcept = default;
};

template<typename InteractionType>
struct InteractionWrapper;

template<typename InteractionType>
auto make_standalone( InteractionWrapper<InteractionType> & interaction ) noexcept
    -> std::unique_ptr<StandaloneAdaptor<typename InteractionType::state_t>>;

template<typename InteractionType, typename IndexVector>
auto make_standalone( InteractionWrapper<InteractionType> & interaction, const IndexVector & indices ) noexcept
    -> std::unique_ptr<StandaloneAdaptor<typename InteractionType::state_t>>;

template<typename InteractionType>
class StandaloneAdaptor_NonLocal final : public StandaloneAdaptor<typename InteractionType::state_t>
{
    static_assert(
        !is_local<InteractionType>::value, "interaction type for non-local standalone adaptor must be non-local" );

    using Interaction = InteractionType;
    using Data        = typename InteractionType::Data;
    using Cache       = typename InteractionType::Cache;

    // private constructor tag with factory function (for std::unique_ptr) declared as friend
    // to make this the only way to instanciate this object
    struct constructor_tag
    {
        explicit constructor_tag() = default;
    };

public:
    using state_t = typename InteractionType::state_t;

    template<typename T>
    friend auto make_standalone( InteractionWrapper<T> & interaction ) noexcept
        -> std::unique_ptr<StandaloneAdaptor<typename T::state_t>>;

    constexpr StandaloneAdaptor_NonLocal( constructor_tag, const Data & data, Cache & cache ) noexcept
            : StandaloneAdaptor<state_t>(), data( data ), cache( cache ){};

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

    void Gradient( const state_t & state, vectorfield & gradient ) final
    {
        std::invoke( typename InteractionType::Gradient( data, cache ), state, gradient );
    }

    void Hessian( const state_t & state, MatrixX & hessian ) final
    {
        auto hessian_functor = Functor::dense_hessian_wrapper( hessian );
        std::invoke( typename InteractionType::Hessian( data, cache ), state, hessian_functor );
    }

    std::string_view Name() const final
    {
        return InteractionType::name;
    }

private:
    const Data & data;
    Cache & cache;
};

template<typename InteractionType, typename IndexVector>
class StandaloneAdaptor_Local final : public StandaloneAdaptor<typename InteractionType::state_t>
{
    static_assert( is_local<InteractionType>::value, "interaction type for local standalone adaptor must be local" );

    using Interaction = InteractionType;
    using Data        = typename InteractionType::Data;
    using Cache       = typename InteractionType::Cache;
    using IndexTuple  = typename IndexVector::value_type;

    // private constructor tag with factory function (for std::unique_ptr) declared as friend
    // to make this the only way to instanciate this object
    struct constructor_tag
    {
        explicit constructor_tag() = default;
    };

public:
    using state_t = typename InteractionType::state_t;

    template<typename T, typename V>
    friend auto make_standalone( InteractionWrapper<T> & interaction, const V & indices ) noexcept
        -> std::unique_ptr<StandaloneAdaptor<typename T::state_t>>;

    constexpr StandaloneAdaptor_Local(
        constructor_tag, const Data & data, Cache & cache, const IndexVector & indices ) noexcept
            : StandaloneAdaptor<state_t>(), data( data ), cache( cache ), indices( indices ){};

    scalar Energy( const state_t & state ) final
    {
        using std::begin, std::end;
        auto functor           = typename InteractionType::Energy( data, cache );
        const auto * state_ptr = raw_pointer_cast( state.data() );
        return Backend::transform_reduce(
            SPIRIT_PAR indices.begin(), indices.end(), scalar( 0.0 ), Backend::plus<scalar>{},
            [state_ptr, functor] SPIRIT_HOSTDEVICE( const IndexTuple & index )
            { return functor( Backend::get<typename InteractionType::Index>( index ), state_ptr ); } );
    }

    void Energy_per_Spin( const state_t & state, scalarfield & energy_per_spin ) final
    {
        using std::begin, std::end;
        auto functor           = typename InteractionType::Energy( data, cache );
        const auto * state_ptr = raw_pointer_cast( state.data() );
        Backend::transform(
            SPIRIT_PAR indices.begin(), indices.end(), energy_per_spin.begin(),
            [state_ptr, functor] SPIRIT_HOSTDEVICE( const IndexTuple & index )
            { return functor( Backend::get<typename InteractionType::Index>( index ), state_ptr ); } );
    }

    scalar Energy_Single_Spin( const int ispin, const state_t & state ) final
    {
        return std::invoke(
            typename InteractionType::Energy( data, cache ),
            Backend::get<typename InteractionType::Index>( indices[ispin] ), raw_pointer_cast( state.data() ) );
    }

    void Gradient( const state_t & state, vectorfield & gradient ) final
    {
        using std::begin, std::end;
        auto functor           = typename InteractionType::Gradient( data, cache );
        const auto * state_ptr = raw_pointer_cast( state.data() );
        Backend::transform(
            SPIRIT_PAR indices.begin(), indices.end(), gradient.begin(),
            [state_ptr, functor] SPIRIT_HOSTDEVICE( const IndexTuple & index )
            { return functor( Backend::get<typename InteractionType::Index>( index ), state_ptr ); } );
    }

    void Hessian( const state_t & state, MatrixX & hessian ) final
    {
        using std::begin, std::end;
        auto functor = typename InteractionType::Hessian( data, cache );
        Backend::cpu::for_each(
            indices.begin(), indices.end(),
            [&state, &functor, hessian_functor = Functor::dense_hessian_wrapper( hessian )]( const IndexTuple & index )
            { functor( Backend::get<typename InteractionType::Index>( index ), state, hessian_functor ); } );
    }

    std::string_view Name() const final
    {
        return InteractionType::name;
    }

private:
    // all member variables must be read only to avoid race conditions in the parallelized Backend
    const Data & data;
    const Cache & cache;
    const IndexVector & indices;
};

template<typename InteractionType>
auto make_standalone( InteractionWrapper<InteractionType> & interaction ) noexcept
    -> std::unique_ptr<StandaloneAdaptor<typename InteractionType::state_t>>
{
    static_assert(
        !is_local<InteractionType>::value, "interaction type for non-local standalone adaptor must be non-local" );
    using T = StandaloneAdaptor_NonLocal<InteractionType>;
    return std::make_unique<T>( typename T::constructor_tag{}, interaction.data, interaction.cache );
};

template<typename InteractionType, typename IndexVector>
auto make_standalone( InteractionWrapper<InteractionType> & interaction, const IndexVector & indices ) noexcept
    -> std::unique_ptr<StandaloneAdaptor<typename InteractionType::state_t>>
{
    static_assert( is_local<InteractionType>::value, "interaction type for local standalone adaptor must be local" );
    using T = StandaloneAdaptor_Local<InteractionType, IndexVector>;
    return std::make_unique<T>( typename T::constructor_tag{}, interaction.data, interaction.cache, indices );
};

template<typename InteractionType>
struct InteractionWrapper
{
    using Interaction = InteractionType;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;

    static_assert(
        std::is_default_constructible<Cache>::value, "InteractionType::Cache has to be default constructible" );

    template<typename T>
    friend auto make_standalone( InteractionWrapper<T> & interaction ) noexcept
        -> std::unique_ptr<StandaloneAdaptor<typename T::state_t>>;

    template<typename T, typename IndexVector>
    friend auto make_standalone( InteractionWrapper<T> & interaction, const IndexVector & indices ) noexcept
        -> std::unique_ptr<StandaloneAdaptor<typename T::state_t>>;

    explicit InteractionWrapper( typename InteractionType::Data && init_data ) : data( init_data ), cache(){};
    explicit InteractionWrapper( const typename InteractionType::Data & init_data ) : data( init_data ), cache(){};

    template<template<typename> typename FunctorAccessor>
    FunctorAccessor<InteractionType>
    make_functor() noexcept( std::is_nothrow_constructible<FunctorAccessor<InteractionType>, Data, Cache>::value )
    {
        return FunctorAccessor<InteractionType>( data, cache );
    }

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

template<typename... InteractionType, typename Iterator>
constexpr Iterator
generate_active_nonlocal( Backend::tuple<InteractionWrapper<InteractionType>...> & interactions, Iterator iterator )
{
    static_assert(
        std::conjunction<std::negation<is_local<InteractionType>>...>::value,
        "all interaction types in tuple must be non-local" );

    return Backend::apply(
        [&iterator]( InteractionWrapper<InteractionType> &... elements )
        {
            ( ...,
              [&iterator]( InteractionWrapper<InteractionType> & interaction )
              {
                  if( interaction.is_contributing() )
                      *( iterator++ ) = make_standalone( interaction );
              }( elements ) );

            return iterator;
        },
        interactions );
};

template<typename... InteractionType, typename IndexVector, typename Iterator>
constexpr Iterator generate_active_local(
    Backend::tuple<InteractionWrapper<InteractionType>...> & interactions, const IndexVector & indices,
    Iterator iterator )
{
    static_assert(
        std::conjunction<is_local<InteractionType>...>::value, "all interaction types in tuple must be local" );

    return Backend::apply(
        [&indices, &iterator]( InteractionWrapper<InteractionType> &... elements )
        {
            ( ...,
              [&indices, &iterator]( InteractionWrapper<InteractionType> & interaction )
              {
                  if( interaction.is_contributing() )
                      *( iterator++ ) = make_standalone( interaction, indices );
              }( elements ) );

            return iterator;
        },
        interactions );
};

} // namespace Interaction

} // namespace Spin

} // namespace Engine
