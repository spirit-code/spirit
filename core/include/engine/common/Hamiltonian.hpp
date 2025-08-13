#pragma once
#ifndef SPIRIT_CORE_ENGINE_COMMON_HAMILTONIAN_HPP
#define SPIRIT_CORE_ENGINE_COMMON_HAMILTONIAN_HPP

#include <Spirit/Hamiltonian.h>
#include <Spirit/Spirit_Defines.h>
#include <data/Geometry.hpp>
#include <data/Misc.hpp>
#include <engine/Backend.hpp>
#include <engine/FFT.hpp>
#include <engine/Index_Container.hpp>
#include <engine/Span.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Vectormath_Defines.hpp>
#include <engine/common/Interaction_Wrapper.hpp>
#include <utility/Variadic_Traits.hpp>

#include <memory>

namespace Engine
{

namespace Common
{

// Hamiltonian for (pure) spin systems
template<
    typename InteractionInterfaceType, template<typename, typename> typename InteractionWrapperTemplate,
    typename... InteractionTypes>
struct Hamiltonian
{
public:
    using state_t              = typename InteractionInterfaceType::state_t;
    using InteractionInterface = InteractionInterfaceType;
    template<typename Interaction>
    using InteractionWrapper = InteractionWrapperTemplate<Interaction, InteractionInterface>;

    static_assert( std::conjunction<std::is_same<state_t, typename InteractionTypes::state_t>...>::value );
    using InteractionTuple = Backend::tuple<InteractionWrapper<InteractionTypes>...>;

    template<typename... DataTypes>
    Hamiltonian( Data::Geometry geometry, DataTypes &&... data )
            : geometry( std::make_shared<::Data::Geometry>( std::move( geometry ) ) ),
              interactions( InteractionWrapper<InteractionTypes>( data )... )
    {
        applyGeometry();
    };

    Hamiltonian( Data::Geometry geometry )
            : geometry( std::make_shared<::Data::Geometry>( std::move( geometry ) ) ),
              interactions( InteractionWrapper<InteractionTypes>()... )
    {
        applyGeometry();
    };

    // rule of five, because we use pointers to the geometry and the boundary_conditions in the cache
    // this choice should keep the interfaces a bit cleaner and allow adding more global dependencies
    // in the future.
    ~Hamiltonian() = default;
    Hamiltonian( const Hamiltonian & other ) : geometry( other.geometry ), interactions( other.interactions )
    {
        setPtrAddress();
    };
    Hamiltonian & operator=( const Hamiltonian & other )
    {
        if( this != &other )
        {
            geometry     = other.geometry;
            interactions = other.interactions;
            setPtrAddress();
        };
        return *this;
    };
    Hamiltonian( Hamiltonian && other ) noexcept
            : geometry( std::move( other.geometry ) ), interactions( std::move( other.interactions ) )
    {
        setPtrAddress();
    };
    Hamiltonian & operator=( Hamiltonian && other ) noexcept
    {
        if( this != &other )
        {
            geometry     = std::move( other.geometry );
            interactions = std::move( other.interactions );
            setPtrAddress();
        };
        return *this;
    };

    void Energy_per_Spin( const state_t & state, scalarfield & energy_per_spin )
    {
        if( energy_per_spin.size() != geometry->nos )
            energy_per_spin = scalarfield( geometry->nos, 0.0 );
        else
            Vectormath::fill( energy_per_spin, 0.0 );

        Backend::apply(
            [&state, &energy_per_spin]( auto &... interaction )
            { ( ..., interaction.Energy_per_Spin( state, energy_per_spin ) ); }, interactions );
    };

    void Energy_Contributions_per_Spin( const state_t & state, Data::vectorlabeled<scalarfield> & contributions )
    {
        auto active           = active_interactions();
        const auto & n_active = active.size();

        if( contributions.size() != n_active )
        {
            contributions = Data::vectorlabeled<scalarfield>( n_active, { "", scalarfield( geometry->nos, 0.0 ) } );
        }

        Backend::cpu::transform(
            active.begin(), active.end(), contributions.begin(),
            [&state, nos = geometry->nos]( InteractionInterface * interaction )
            {
                scalarfield energy_per_spin( nos, 0.0 );
                interaction->Energy_per_Spin( state, energy_per_spin );
                return std::make_pair( interaction->Name(), energy_per_spin );
            } );
    };

    [[nodiscard]] Data::vectorlabeled<scalar> Energy_Contributions( const state_t & state )
    {
        auto active           = active_interactions();
        const auto & n_active = active.size();
        Data::vectorlabeled<scalar> contributions( n_active, { "", 0.0 } );

        Backend::cpu::transform(
            active.begin(), active.end(), contributions.begin(), [&state]( InteractionInterface * interaction )
            { return std::make_pair( interaction->Name(), interaction->Energy( state ) ); } );

        return contributions;
    };

    // Calculate the total energy for a single spin to be used in Monte Carlo.
    //      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
    [[nodiscard]] scalar Energy_Single_Spin( const int ispin, const state_t & state )
    {
        return Backend::apply(
            [ispin, &state]( auto &... interaction )
            { return ( scalar( 0 ) + ... + interaction.Energy_Single_Spin( ispin, state ) ); }, interactions );
    };

    [[nodiscard]] scalar Energy( const state_t & state )
    {
        return Backend::apply(
            [&state]( auto &... interaction ) { return ( scalar( 0 ) + ... + interaction.Energy( state ) ); },
            interactions );
    };

    [[nodiscard]] static std::size_t total_count()
    {
        return std::tuple_size_v<InteractionTuple>;
    }

    [[nodiscard]] auto total_interactions() -> std::vector<InteractionInterface *>
    {
        return Backend::apply(
            []( auto &... interaction ) -> std::vector<InteractionInterface *>
            { return { std::addressof( interaction )... }; }, interactions );
    }

    [[nodiscard]] auto total_interactions() const -> std::vector<const InteractionInterface *>
    {
        return Backend::apply(
            []( const auto &... interaction ) -> std::vector<const InteractionInterface *>
            { return { std::addressof( interaction )... }; }, interactions );
    }

    [[nodiscard]] std::size_t active_count() const
    {
        return Backend::apply(
            []( const auto &... interaction )
            { return ( std::size_t( 0 ) + ... + ( interaction.is_contributing() ? 1 : 0 ) ); }, interactions );
    }

    [[nodiscard]] auto active_interactions() -> std::vector<InteractionInterface *>
    {

        return Backend::apply(
            []( auto &... interaction )
            {
                auto active = std::vector<InteractionInterface *>( 0 );
                ( ..., ( interaction.is_contributing() ? active.push_back( &interaction ) : void() ) );
                return active;
            },
            interactions );
    };

    [[nodiscard]] auto active_interactions() const -> std::vector<const InteractionInterface *>
    {
        return Backend::apply(
            []( const auto &... interaction )
            {
                auto active = std::vector<const InteractionInterface *>( 0 );
                ( ..., ( interaction.is_contributing() ? active.push_back( &interaction ) : void() ) );
                return active;
            },
            interactions );
    };

    // compile time getter
    template<class T>
    [[nodiscard]] constexpr auto getInteraction() -> InteractionInterface *
    {
        if constexpr( hasInteraction<T>() )
            return &Backend::get<InteractionWrapper<T>>( interactions );
        else
            return nullptr;
    };

    // compile time getter
    template<class T>
    [[nodiscard]] constexpr auto getInteraction() const -> const InteractionInterface *
    {
        if constexpr( Utility::contains<InteractionWrapper<T>, InteractionTuple>::value )
            return &Backend::get<InteractionWrapper<T>>( interactions );
        else
            return nullptr;
    };

    template<class T>
    [[nodiscard]] static constexpr bool hasInteraction()
    {
        return Utility::contains<InteractionWrapper<T>, InteractionTuple>::value;
    };

    void setPtrAddress() noexcept
    {
        Backend::apply(
            [geometry = geometry.get()]( InteractionWrapper<InteractionTypes> &... interaction )
            { ( ..., interaction.set_ptr_address( geometry ) ); }, interactions );
    }

    void applyGeometry()
    {
        if( geometry == nullptr )
        {
            setPtrAddress();
            return;
        }

        Backend::apply(
            [this]( auto &... interaction ) { ( ..., interaction.applyGeometry( *geometry ) ); }, interactions );
    }

    template<typename T>
    [[nodiscard]] auto data() const -> const typename T::Data *
    {
        if constexpr( hasInteraction<T>() )
            return &Backend::get<InteractionWrapper<T>>( interactions ).get_data();
        else
            return nullptr;
    };

    template<typename T>
    [[nodiscard]] auto set_data( typename T::Data && data ) -> std::optional<std::string>
    {
        std::optional<std::string> error{};

        if constexpr( hasInteraction<T>() )
        {
            error = Backend::get<InteractionWrapper<T>>( interactions ).set_data( std::move( data ) );
            applyGeometry<T>();
        }
        else
            error = fmt::format( "The Hamiltonian doesn't contain an interaction of type \"{}\"", T::name );

        return error;
    };

    template<typename T, typename... Args>
    [[nodiscard]] auto set_data( Args &&... args ) -> std::optional<std::string>
    {
        return set_data<T>( typename T::Data{ std::forward<Args>( args )... } );
    }

    template<typename T>
    [[nodiscard]] auto cache() const -> const typename T::Cache *
    {
        if constexpr( hasInteraction<T>() )
            return &Backend::get<InteractionWrapper<T>>( interactions ).get_cache();
        else
            return nullptr;
    };

    template<class T>
    [[nodiscard]] bool is_contributing() const
    {
        if constexpr( hasInteraction<T>() )
            return getInteraction<T>()->is_contributing();
        else
            return false;
    }

    [[nodiscard]] const auto & get_geometry() const
    {
        return *geometry;
    }

    void set_geometry( const ::Data::Geometry & g )
    {
        set_geometry_impl( g );
    }

    void set_geometry( ::Data::Geometry && g )
    {
        set_geometry_impl( std::move( g ) );
    }

    // TODO: eliminate the need for these
    // [[nodiscard]] const auto & get_boundary_conditions() const
    // {
    //     return geometry->boundary_conditions;
    // }
    //
    void set_boundary_conditions( const intfield & bc )
    {
        geometry->boundary_conditions = bc;
        applyGeometry();
    }

private:
    template<typename InteractionType>
    void applyGeometry()
    {
        static_assert( hasInteraction<InteractionType>() );

        if( geometry == nullptr )
        {
            setPtrAddress();
            return;
        }

        Backend::get<InteractionWrapper<InteractionType>>( interactions ).applyGeometry( *geometry );
    };

    template<typename Geometry>
    void set_geometry_impl( Geometry && g )
    {
        static_assert( std::is_same_v<std::decay_t<Geometry>, ::Data::Geometry> );
        // lazy copy mechanism for the geometry
        // We allow shallow copies when the geometry stays the same,
        // but if we want to change it we ensure that we are the sole owner of the Geometry
        // This only works, because the Geometry class is only shared between Hamiltonian objects
        if( geometry.use_count() > 1 || geometry == nullptr )
        {
            geometry = std::make_shared<::Data::Geometry>( std::forward<Geometry>( g ) );
        }
        else
        {
            *geometry = std::forward<Geometry>( g );
        }
        applyGeometry();
    }

protected:
    InteractionTuple interactions;

private:
    std::shared_ptr<Data::Geometry> geometry;
};

} // namespace Common

} // namespace Engine

#endif
