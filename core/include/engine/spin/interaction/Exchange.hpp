#pragma once
#ifndef SPIRIT_CORE_ENGINE_INTERACTION_EXCHANGE_HPP
#define SPIRIT_CORE_ENGINE_INTERACTION_EXCHANGE_HPP

#include <engine/Indexing.hpp>
#include <engine/Neighbours.hpp>
#include <engine/Span.hpp>
#include <engine/spin/interaction/Functor_Prototypes.hpp>

#include <Eigen/Dense>

namespace Engine
{

namespace Spin
{

namespace Interaction
{

struct Exchange
{
    using state_t = vectorfield;

    struct Data
    {
        pairfield pairs{};
        scalarfield magnitudes{};

        scalarfield shell_magnitudes{};

        Data() = default;
        Data( pairfield pairs, scalarfield magnitudes )
                : pairs( std::move( pairs ) ), magnitudes( std::move( magnitudes ) ) {};

        Data( scalarfield shell_magnitudes ) : shell_magnitudes( std::move( shell_magnitudes ) ) {};
    };

    static bool valid_data( const Data & data )
    {
        if( !data.shell_magnitudes.empty() )
            return true;
        else
            return data.pairs.empty() || ( data.pairs.size() == data.magnitudes.size() );
    };

    struct Cache
    {
        pairfield pairs{};
        scalarfield magnitudes{};
    };

    static bool is_contributing( const Data &, const Cache & cache )
    {
        return !cache.pairs.empty();
    }

    struct IndexType
    {
        int ispin, jspin, ipair;
    };

    using Index        = Engine::Span<const IndexType>;
    using IndexStorage = Backend::vector<IndexType>;

    using Energy   = Functor::Local::Energy_Functor<Functor::Local::DataRef<Exchange>>;
    using Gradient = Functor::Local::Gradient_Functor<Functor::Local::DataRef<Exchange>>;
    using Hessian  = Functor::Local::Hessian_Functor<Functor::Local::DataRef<Exchange>>;

    static std::size_t Sparse_Hessian_Size_per_Cell( const Data &, const Cache & cache )
    {
        return cache.pairs.size() * 2;
    };

    // Calculate the total energy for a single spin to be used in Monte Carlo.
    //      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
    using Energy_Single_Spin = Functor::Local::Energy_Single_Spin_Functor<Energy, 2>;

    // Interaction name as string
    static constexpr std::string_view name = "Exchange";

    template<typename IndexStorageVector>
    static void applyGeometry(
        const ::Data::Geometry & geometry, const intfield & boundary_conditions, const Data & data, Cache & cache,
        IndexStorageVector & indices )
    {
        using Indexing::idx_from_pair;

        // redundant neighbours are captured when expanding pairs below
        static constexpr bool use_redundant_neighbours = false;

        cache.pairs      = pairfield( 0 );
        cache.magnitudes = scalarfield( 0 );
        if( !data.shell_magnitudes.empty() )
        {
            // Generate Exchange neighbours
            intfield exchange_shells( 0 );
            Neighbours::Get_Neighbours_in_Shells(
                geometry, data.shell_magnitudes.size(), cache.pairs, exchange_shells, use_redundant_neighbours );
            cache.magnitudes.reserve( cache.pairs.size() );
            for( std::size_t ipair = 0; ipair < cache.pairs.size(); ++ipair )
            {
                cache.magnitudes.push_back( data.shell_magnitudes[exchange_shells[ipair]] );
            }
        }
        else
        {
            // Use direct list of pairs
            cache.pairs      = data.pairs;
            cache.magnitudes = data.magnitudes;
        }

        for( unsigned int icell = 0; icell < geometry.n_cells_total; ++icell )
        {
            for( unsigned int i_pair = 0; i_pair < cache.pairs.size(); ++i_pair )
            {
                int ispin = cache.pairs[i_pair].i + icell * geometry.n_cell_atoms;
                int jspin = idx_from_pair(
                    ispin, boundary_conditions, geometry.n_cells, geometry.n_cell_atoms, geometry.atom_types,
                    cache.pairs[i_pair] );
                if( jspin >= 0 )
                {
                    Backend::get<IndexStorage>( indices[ispin] ).push_back( IndexType{ ispin, jspin, (int)i_pair } );
                    Backend::get<IndexStorage>( indices[jspin] ).push_back( IndexType{ jspin, ispin, (int)i_pair } );
                }
            }
        }
    };
};

template<>
struct Functor::Local::DataRef<Exchange>
{
    using Interaction = Exchange;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;

    DataRef( const Data & data, const Cache & cache ) noexcept
            : is_contributing( Interaction::is_contributing( data, cache ) ), magnitudes( cache.magnitudes.data() )
    {
    }

    const bool is_contributing;

protected:
    const scalar * magnitudes;
};

template<>
inline scalar Exchange::Energy::operator()( const Index & index, const Vector3 * spins ) const
{
    // don't need to check for `is_contributing` here, because the `transform_reduce` will short circuit correctly
    return Backend::transform_reduce(
        index.begin(), index.end(), scalar( 0.0 ), Backend::plus<scalar>{},
        [this, spins] SPIRIT_LAMBDA( const Exchange::IndexType & idx ) -> scalar
        {
            const auto & [ispin, jspin, i_pair] = idx;
            return -0.5 * magnitudes[i_pair] * spins[ispin].dot( spins[jspin] );
        } );
}

template<>
inline Vector3 Exchange::Gradient::operator()( const Index & index, const Vector3 * spins ) const
{
    // don't need to check for `is_contributing` here, because the `transform_reduce` will short circuit correctly
    return Backend::transform_reduce(
        index.begin(), index.end(), Vector3{ 0.0, 0.0, 0.0 }, Backend::plus<Vector3>{},
        [this, spins] SPIRIT_LAMBDA( const Exchange::IndexType & idx ) -> Vector3
        {
            const auto & [ispin, jspin, i_pair] = idx;
            return -magnitudes[i_pair] * spins[jspin];
        } );
}

template<>
template<typename Callable>
void Exchange::Hessian::operator()( const Index & index, const vectorfield &, Callable & hessian ) const
{
    // don't need to check for `is_contributing` here, because `for_each` will short circuit properly
    Backend::cpu::for_each(
        index.begin(), index.end(),
        [this, &index, &hessian]( const Exchange::IndexType & idx )
        {
            const int i         = 3 * idx.ispin;
            const int j         = 3 * idx.jspin;
            const auto & i_pair = idx.ipair;

#pragma unroll
            for( int alpha = 0; alpha < 3; ++alpha )
            {
                hessian( i + alpha, j + alpha, -magnitudes[i_pair] );
            }
        } );
}

} // namespace Interaction

} // namespace Spin

} // namespace Engine

#endif
