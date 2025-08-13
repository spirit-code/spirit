#pragma once
#ifndef SPIRIT_CORE_ENGINE_INTERACTION_QUADRUPLET_HPP
#define SPIRIT_CORE_ENGINE_INTERACTION_QUADRUPLET_HPP

#include <engine/Index_Container.hpp>
#include <engine/Indexing.hpp>
#include <engine/Span.hpp>
#include <engine/spin/StateType.hpp>
#include <engine/spin/interaction/Functor_Prototypes.hpp>

namespace Engine
{

namespace Spin
{

namespace Interaction
{

struct Quadruplet
{
    using state_t = StateType;

    struct Data
    {
        quadrupletfield quadruplets{};
        scalarfield magnitudes{};

        Data() = default;
        Data( quadrupletfield quadruplets, scalarfield magnitudes )
                : quadruplets( std::move( quadruplets ) ), magnitudes( std::move( magnitudes ) ) {};
    };

    static bool valid_data( const Data & data )
    {
        return data.quadruplets.size() == data.magnitudes.size();
    };

    struct Cache
    {
        const ::Data::Geometry * geometry{};
    };

    static bool is_contributing( const Data & data, const Cache & )
    {
        return !data.quadruplets.empty();
    }

    struct Index
    {
        int ispin, jspin, kspin, lspin, iquad;
    };

    using IndexContainer = Engine::IndexContainer<Quadruplet>;

    using Energy   = Functor::Local::Energy_Functor<Functor::Local::DataRef<Quadruplet>>;
    using Gradient = Functor::Local::Gradient_Functor<Functor::Local::DataRef<Quadruplet>>;
    using Hessian  = Functor::Local::Hessian_Functor<Functor::Local::DataRef<Quadruplet>>;

    static std::size_t Sparse_Hessian_Size_per_Cell( const Data &, const Cache & )
    {
        return 0;
    };

    // Calculate the total energy for a single spin to be used in Monte Carlo.
    //      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
    using Energy_Single_Spin = Functor::Local::Energy_Single_Spin_Functor<Energy, 4>;

    // Interaction name as string
    static constexpr std::string_view name = "Quadruplet";

    static constexpr bool local = true;

    static void
    applyGeometry( const ::Data::Geometry & geometry, const Data & data, Cache & cache, IndexContainer & container )
    {
        using Indexing::idx_from_pair;
        auto indices = std::vector( geometry.nos, field<Index>{} );

        for( int iquad = 0; iquad < data.quadruplets.size(); ++iquad )
        {
            const auto & quad = data.quadruplets[iquad];

            const int i = quad.i;
            const int j = quad.j;
            const int k = quad.k;
            const int l = quad.l;

            const auto & d_j = quad.d_j;
            const auto & d_k = quad.d_k;
            const auto & d_l = quad.d_l;

            const auto & boundary_conditions = geometry.boundary_conditions;
            for( unsigned int icell = 0; icell < geometry.n_cells_total; ++icell )
            {
                int ispin = i + icell * geometry.n_cell_atoms;
                int jspin = idx_from_pair(
                    ispin, boundary_conditions, geometry.n_cells, geometry.n_cell_atoms, geometry.atom_types,
                    { i, j, d_j } );
                int kspin = idx_from_pair(
                    ispin, boundary_conditions, geometry.n_cells, geometry.n_cell_atoms, geometry.atom_types,
                    { i, k, d_k } );
                int lspin = idx_from_pair(
                    ispin, boundary_conditions, geometry.n_cells, geometry.n_cell_atoms, geometry.atom_types,
                    { i, l, d_l } );

                if( jspin < 0 || kspin < 0 || lspin < 0 )
                    continue;

                indices[ispin].push_back( Index{ ispin, jspin, kspin, lspin, (int)iquad } );
                indices[jspin].push_back( Index{ jspin, ispin, kspin, lspin, (int)iquad } );
                indices[kspin].push_back( Index{ kspin, lspin, ispin, jspin, (int)iquad } );
                indices[lspin].push_back( Index{ lspin, kspin, ispin, jspin, (int)iquad } );
            }
        }

        container      = make_index_container<Quadruplet>( std::move( indices ) );
        cache.geometry = &geometry;
    };
};

template<>
struct Functor::Local::DataRef<Quadruplet>
{
    using Interaction = Quadruplet;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;

    DataRef( const Data & data, const Cache & cache ) noexcept
            : is_contributing( Interaction::is_contributing( data, cache ) ), magnitudes( data.magnitudes.data() )
    {
    }

    const bool is_contributing;

protected:
    const scalar * magnitudes;
};

template<>
inline scalar Quadruplet::Energy::operator()( Span<const Index> index, quantity<const Vector3 *> state ) const
{
    // don't need to check for `is_contributing` here, because the `transform_reduce` will short circuit correctly
    return Backend::transform_reduce(
        index.begin(), index.end(), scalar( 0.0 ), Backend::plus<scalar>{},
        [this, state] SPIRIT_LAMBDA( const Index & idx ) -> scalar
        {
            const auto & [ispin, jspin, kspin, lspin, iquad] = idx;
            return -0.25 * magnitudes[iquad] * ( state.spin[ispin].dot( state.spin[jspin] ) )
                   * ( state.spin[kspin].dot( state.spin[lspin] ) );
        } );
}

template<>
inline Vector3 Quadruplet::Gradient::operator()( Span<const Index> index, quantity<const Vector3 *> state ) const
{
    // don't need to check for `is_contributing` here, because the `transform_reduce` will short circuit correctly
    return Backend::transform_reduce(
        index.begin(), index.end(), Vector3{ 0.0, 0.0, 0.0 }, Backend::plus<Vector3>{},
        [this, state] SPIRIT_LAMBDA( const Index & idx ) -> Vector3
        {
            const auto & [ispin, jspin, kspin, lspin, iquad] = idx;
            return state.spin[jspin] * ( -magnitudes[iquad] * ( state.spin[kspin].dot( state.spin[lspin] ) ) );
        } );
}

template<>
template<typename Callable>
void Quadruplet::Hessian::operator()( Span<const Index> index, const StateType & state, Callable & hessian ) const
{
    Backend::cpu::for_each(
        index.begin(), index.end(),
        [this, &index, &state, &hessian]( const Index & idx )
        {
            const auto & [ispin, jspin, kspin, lspin, iquad] = idx;

            for( int alpha = 0; alpha < 3; ++alpha )
            {
                hessian(
                    3 * ispin + alpha, 3 * jspin + alpha,
                    -magnitudes[iquad] * state.spin[kspin].dot( state.spin[lspin] ) );
                for( int beta = 0; beta < 3; ++beta )
                {
                    hessian(
                        3 * ispin + alpha, 3 * kspin + beta,
                        -magnitudes[iquad] * state.spin[jspin][alpha] * state.spin[lspin][beta] );
                    hessian(
                        3 * ispin + alpha, 3 * lspin + beta,
                        -magnitudes[iquad] * state.spin[jspin][alpha] * state.spin[kspin][beta] );
                }
            }
        } );
};

} // namespace Interaction

} // namespace Spin

} // namespace Engine

#endif
