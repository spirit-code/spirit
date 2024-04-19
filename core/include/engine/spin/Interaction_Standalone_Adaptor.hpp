#pragma once

#include <engine/common/Interaction_Standalone_Adaptor.hpp>
#include <engine/common/Interaction_Wrapper.hpp>
#include <engine/spin/interaction/Functor_Prototypes.hpp>

namespace Engine
{

namespace Spin
{

namespace Interaction
{

using Common::Interaction::InteractionWrapper;
using Common::Interaction::is_local;
template<typename state_type>
struct StandaloneAdaptor : public Common::Interaction::StandaloneAdaptor<state_type>
{
    using state_t = state_type;

    virtual void Gradient( const state_t & state, vectorfield & gradient ) = 0;
    virtual void Hessian( const state_t & state, MatrixX & hessian )       = 0;

protected:
    constexpr StandaloneAdaptor() = default;
};

template<typename InteractionType>
class StandaloneAdaptor_NonLocal
        : public Common::Interaction::StandaloneAdaptor_NonLocal<
              InteractionType, Spin::Interaction::StandaloneAdaptor<typename InteractionType::state_t>>
{
    static_assert(
        !is_local<InteractionType>::value, "interaction type for non-local standalone adaptor must be non-local" );

    using base_t = Common::Interaction::StandaloneAdaptor_NonLocal<
        InteractionType, Spin::Interaction::StandaloneAdaptor<typename InteractionType::state_t>>;

    using Interaction = InteractionType;
    using Data        = typename InteractionType::Data;
    using Cache       = typename InteractionType::Cache;

    struct constructor_tag
    {
        explicit constructor_tag() = default;
    };

public:
    using state_t = typename InteractionType::state_t;

    template<typename AdaptorType>
    friend class Common::Interaction::StandaloneFactory;

    StandaloneAdaptor_NonLocal( constructor_tag, const Data & data, Cache & cache ) noexcept : base_t( data, cache ){};

    void Gradient( const state_t & state, vectorfield & gradient ) final
    {
        std::invoke( typename InteractionType::Gradient( this->data, this->cache ), state, gradient );
    }

    void Hessian( const state_t & state, MatrixX & hessian ) final
    {
        auto hessian_functor = Functor::dense_hessian_wrapper( hessian );
        std::invoke( typename InteractionType::Hessian( this->data, this->cache ), state, hessian_functor );
    }
};

template<typename InteractionType, typename IndexVector>
class StandaloneAdaptor_Local : public Common::Interaction::StandaloneAdaptor_Local<
                                    InteractionType, StandaloneAdaptor<typename InteractionType::state_t>, IndexVector>
{
    static_assert( is_local<InteractionType>::value, "interaction type for local standalone adaptor must be local" );

    using base_t = Common::Interaction::StandaloneAdaptor_Local<
        InteractionType, StandaloneAdaptor<typename InteractionType::state_t>, IndexVector>;

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

    template<typename AdaptorType>
    friend class Common::Interaction::StandaloneFactory;

    StandaloneAdaptor_Local( constructor_tag, const Data & data, Cache & cache, const IndexVector & indices ) noexcept
            : base_t( data, cache, indices ){};

    void Gradient( const state_t & state, vectorfield & gradient ) final
    {
        using std::begin, std::end;
        auto functor           = typename InteractionType::Gradient( this->data, this->cache );
        const auto * state_ptr = raw_pointer_cast( state.data() );
        Backend::transform(
            SPIRIT_PAR this->indices.begin(), this->indices.end(), gradient.begin(),
            [state_ptr, functor] SPIRIT_LAMBDA( const IndexTuple & index )
            { return functor( Backend::get<typename InteractionType::Index>( index ), state_ptr ); } );
    }

    void Hessian( const state_t & state, MatrixX & hessian ) final
    {
        using std::begin, std::end;
        auto functor = typename InteractionType::Hessian( this->data, this->cache );
        Backend::cpu::for_each(
            this->indices.begin(), this->indices.end(),
            [&state, &functor, hessian_functor = Functor::dense_hessian_wrapper( hessian )]( const IndexTuple & index )
            { functor( Backend::get<typename InteractionType::Index>( index ), state, hessian_functor ); } );
    }
};

} // namespace Interaction

} // namespace Spin

namespace Common
{

namespace Interaction
{

template<typename state_t>
class StandaloneFactory<Spin::Interaction::StandaloneAdaptor<state_t>>
{
    using AdaptorType = Spin::Interaction::StandaloneAdaptor<state_t>;

public:
    constexpr StandaloneFactory() = default;

    template<typename InteractionType>
    static auto make_standalone( InteractionWrapper<InteractionType> & interaction ) noexcept
        -> std::unique_ptr<AdaptorType>
    {
        static_assert(
            !is_local<InteractionType>::value, "interaction type for non-local standalone adaptor must be non-local" );
        using T = Spin::Interaction::StandaloneAdaptor_NonLocal<InteractionType>;
        return std::make_unique<T>( typename T::constructor_tag{}, interaction.data, interaction.cache );
    };

    template<typename InteractionType, typename IndexVector>
    static auto
    make_standalone( InteractionWrapper<InteractionType> & interaction, const IndexVector & indices ) noexcept
        -> std::unique_ptr<AdaptorType>
    {
        static_assert(
            is_local<InteractionType>::value, "interaction type for local standalone adaptor must be local" );
        using T = Spin::Interaction::StandaloneAdaptor_Local<InteractionType, IndexVector>;
        return std::make_unique<T>( typename T::constructor_tag{}, interaction.data, interaction.cache, indices );
    };
};

} // namespace Interaction

} // namespace Common

} // namespace Engine
