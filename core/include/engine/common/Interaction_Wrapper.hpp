#pragma once

#include <data/Geometry.hpp>
#include <engine/Index_Container.hpp>
#include <engine/StateType.hpp>
#include <engine/Vectormath_Defines.hpp>
#include <engine/common/Interaction_Traits.hpp>
#include <engine/common/interaction/Functor_Prototypes.hpp>
#include <utility/Exception.hpp>

#include <optional>

namespace Engine
{

namespace Common
{

namespace Interaction
{

template<typename state_type>
struct IWrapper
{
    virtual ~IWrapper() = default;

    using state_t = state_type;

    virtual scalar Energy( const state_t & state )                                       = 0;
    virtual void Energy_per_Spin( const state_t & state, scalarfield & energy_per_spin ) = 0;
    virtual scalar Energy_Single_Spin( int ispin, const state_t & state )                = 0;
    virtual std::string_view Name() const                                                = 0;
    virtual bool is_contributing() const                                                 = 0;

protected:
    constexpr IWrapper() = default;
};

template<typename InteractionType, typename Adaptor>
struct Wrapper;

template<typename InteractionType, typename Adaptor>
struct Wrapper : public Adaptor
{
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
    using Interaction    = InteractionType;
    using Data           = typename Interaction::Data;
    using Cache          = typename Interaction::Cache;
    using IndexContainer = typename Interaction::IndexContainer;

    using state_t = typename Adaptor::state_t;

    static_assert(
        std::is_default_constructible<Cache>::value, "InteractionType::Cache has to be default constructible" );
    static_assert(
        std::is_default_constructible<Data>::value, "InteractionType::Data has to be default constructible" );

    constexpr Wrapper() = default;
    explicit Wrapper( typename InteractionType::Data && init_data )
            : Adaptor(), data( std::move( init_data ) ), cache() {};
    explicit Wrapper( const typename InteractionType::Data & init_data ) : Adaptor(), data( init_data ), cache() {};

    // applyGeometry
    void applyGeometry( const ::Data::Geometry & geometry )
    {
        if constexpr( is_local<Interaction>::value )
        {
            Interaction::applyGeometry( geometry, data, cache, indices );
            if( !Engine::verify_index_container( indices ) )
                spirit_throw(
                    Utility::Exception_Classifier::Standard_Exception, Utility::Log_Level::Error,
                    fmt::format( "Invalid indices set on interaction: '{}'", this->Name() ) );
        }
        else
            Interaction::applyGeometry( geometry, data, cache );
    }

    // is_contributing
    bool is_contributing() const final
    {
        return Interaction::is_contributing( data, cache );
    }

    // set_data
    auto set_data( typename Interaction::Data && new_data ) -> std::optional<std::string>
    {
        if constexpr( has_valid_check<Interaction>::value )
            if( !Interaction::valid_data( new_data ) )
                return { fmt::format( "the data passed to interaction \"{}\" is invalid", Interaction::name ) };

        data = std::move( new_data );
        return std::nullopt;
    };

    auto get_data() const -> const Data &
    {
        return data;
    }

    auto get_cache() const -> const Cache &
    {
        return cache;
    }

    void set_ptr_address( ::Data::Geometry * geometry )
    {
        if constexpr( Common::Interaction::has_geometry_member<Cache>::value )
            cache.geometry = geometry;
    }

    scalar Energy( const state_t & state ) final
    {
        if constexpr( is_local<Interaction>::value )
        {
            if( this->indices.offsets.size() <= 1 )
                return 0;

            const int n_spans = this->indices.offsets.size() - 1;

            auto state_ptr          = static_cast<typename state_traits<state_t>::const_pointer>( state.data() );
            auto functor            = typename Interaction::Energy( data, cache );
            const auto * idx_offset = indices.offsets.data();
            const auto * idx_data   = indices.data.data();

            return Backend::transform_reduce(
                SPIRIT_PAR Backend::make_counting_iterator<int>( 0 ), Backend::make_counting_iterator<int>( n_spans ),
                scalar( 0.0 ), Backend::plus<scalar>{},
                [state_ptr, functor, idx_offset, idx_data] SPIRIT_LAMBDA( const int idx ) {
                    return functor(
                        Span( idx_data + idx_offset[idx], idx_offset[idx + 1] - idx_offset[idx] ), state_ptr );
                } );
        }
        else
        {
            return std::invoke( typename Interaction::Energy_Total( data, cache ), state );
        }
    }
    void Energy_per_Spin( const state_t & state, scalarfield & energy_per_spin ) final
    {
        if constexpr( is_local<Interaction>::value )
        {
            if( this->indices.offsets.size() <= 1 )
                return;

            const int n_spans = this->indices.offsets.size() - 1;

            if( energy_per_spin.size() != n_spans )
                spirit_throw(
                    Utility::Exception_Classifier::Standard_Exception, Utility::Log_Level::Error,
                    fmt::format(
                        "Mismatched size for indices in Energy caclulation (Interaction: '{}')", this->Name() ) );

            auto state_ptr          = static_cast<typename state_traits<state_t>::const_pointer>( state.data() );
            auto functor            = typename Interaction::Energy( data, cache );
            const auto * idx_offset = indices.offsets.data();
            const auto * idx_data   = indices.data.data();
            auto * energy           = energy_per_spin.data();

            Backend::for_each_n(
                SPIRIT_PAR Backend::make_counting_iterator<int>( 0 ), n_spans,
                [state_ptr, functor, idx_offset, idx_data, energy] SPIRIT_LAMBDA( const int idx ) {
                    energy[idx] += functor(
                        Span( idx_data + idx_offset[idx], idx_offset[idx + 1] - idx_offset[idx] ), state_ptr );
                } );
        }
        else
        {
            std::invoke( typename Interaction::Energy( data, cache ), state, energy_per_spin );
        }
    }

    scalar Energy_Single_Spin( const int ispin, const state_t & state ) final
    {
        if constexpr( is_local<Interaction>::value )
        {
            if( this->indices.offsets.size() <= 1 )
                return 0.0;

            return std::invoke(
                typename Interaction::Energy_Single_Spin( data, cache ),
                Span<const typename Interaction::Index>(
                    indices.data.data() + indices.offsets[ispin], indices.offsets[ispin + 1] - indices.offsets[ispin] ),
                state.data() );
        }
        else
        {
            return std::invoke( typename Interaction::Energy_Single_Spin( data, cache ), ispin, state );
        }
    }

    std::string_view Name() const final
    {
        return Interaction::name;
    }

protected:
    Data data              = Data();
    Cache cache            = Cache();
    IndexContainer indices = IndexContainer();
};

} // namespace Interaction

} // namespace Common

} // namespace Engine
