#pragma once
#ifndef SPIRIT_CORE_ENGINE_INTERACTION_ZEEMANN_HPP
#define SPIRIT_CORE_ENGINE_INTERACTION_ZEEMANN_HPP

#include <engine/Index_Container.hpp>
#include <engine/Indexing.hpp>
#include <engine/spin/StateType.hpp>
#include <engine/spin/interaction/Functor_Prototypes.hpp>

namespace Engine
{

namespace Spin
{

namespace Interaction
{

struct Zeeman
{
    using state_t = StateType;

    struct Data
    {
        scalar external_field_magnitude  = 0;
        Vector3 external_field_direction = { 0, 0, 1 };

        Data() = default;
        Data( scalar external_field_magnitude, Vector3 external_field_direction )
                : external_field_magnitude( external_field_magnitude ),
                  external_field_direction( std::move( external_field_direction ) ) {};
    };

    struct Index
    {
        int ispin;
    };

    using IndexContainer = Engine::IndexContainer<Zeeman>;

    struct Cache
    {
        const ::Data::Geometry * geometry;
    };

    static bool is_contributing( const Data & data, const Cache & )
    {
        return std::abs( data.external_field_magnitude ) > 1e-60;
    }

    using Energy   = Functor::Local::Energy_Functor<Functor::Local::DataRef<Zeeman>>;
    using Gradient = Functor::Local::Gradient_Functor<Functor::Local::DataRef<Zeeman>>;
    using Hessian  = Functor::Local::Hessian_Functor<Functor::Local::DataRef<Zeeman>>;

    static std::size_t Sparse_Hessian_Size_per_Cell( const Data &, const Cache & )
    {
        return 0;
    };

    // Calculate the total energy for a single spin to be used in Monte Carlo.
    //      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
    using Energy_Single_Spin = Functor::Local::Energy_Single_Spin_Functor<Energy, 1>;

    // Interaction name as string
    static constexpr std::string_view name = "Zeeman";

    static void
    applyGeometry( const ::Data::Geometry & geometry, const Data &, Cache & cache, IndexContainer & container )
    {
        using Indexing::check_atom_type;
        auto indices = std::vector( geometry.nos, field<Index>{} );

        const auto N = geometry.n_cell_atoms;

        for( int icell = 0; icell < geometry.n_cells_total; ++icell )
        {
            for( int ibasis = 0; ibasis < N; ++ibasis )
            {
                const int ispin = icell * N + ibasis;
                if( check_atom_type( geometry.atom_types[ispin] ) )
                {
                    indices[ispin].push_back( Index{ ispin } );
                }
            };
        }

        container      = make_index_container<Zeeman>( std::move( indices ) );
        cache.geometry = &geometry;
    }
};

template<>
struct Functor::Local::DataRef<Zeeman>
{
    using Interaction = Zeeman;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;

    DataRef( const Data & data, const Cache & cache ) noexcept
            : is_contributing( Interaction::is_contributing( data, cache ) ),
              external_field_magnitude( data.external_field_magnitude ),
              external_field_direction( data.external_field_direction ),
              mu_s( cache.geometry->mu_s.data() )
    {
    }

    const bool is_contributing;

protected:
    const scalar external_field_magnitude;
    const Vector3 external_field_direction;
    const scalar * mu_s;
};

template<>
inline scalar Zeeman::Energy::operator()( Span<const Index> index, quantity<const Vector3 *> state ) const
{
    if( !is_contributing )
        return 0.0;
    else
        return Backend::transform_reduce(
            index.begin(), index.end(), scalar( 0.0 ), Backend::plus<scalar>{},
            [this, state] SPIRIT_LAMBDA( const Index & idx ) -> scalar {
                return -mu_s[idx.ispin] * external_field_magnitude
                       * external_field_direction.dot( state.spin[idx.ispin] );
            } );
}

template<>
inline Vector3 Zeeman::Gradient::operator()( Span<const Index> index, quantity<const Vector3 *> ) const
{
    if( !is_contributing )
        return Vector3::Zero();
    else
        return Backend::transform_reduce(
            index.begin(), index.end(), Vector3{ Vector3::Zero() }, Backend::plus<Vector3>{},
            [this] SPIRIT_LAMBDA( const Index & idx ) -> Vector3
            { return -mu_s[idx.ispin] * external_field_magnitude * external_field_direction; } );
}

template<>
template<typename Callable>
void Zeeman::Hessian::operator()( Span<const Index>, const StateType &, Callable & ) const {};

} // namespace Interaction

} // namespace Spin

} // namespace Engine
#endif
