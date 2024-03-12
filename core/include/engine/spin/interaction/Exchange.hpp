#pragma once
#ifndef SPIRIT_CORE_ENGINE_INTERACTION_EXCHANGE_HPP
#define SPIRIT_CORE_ENGINE_INTERACTION_EXCHANGE_HPP

#include <engine/Indexing.hpp>
#include <engine/Neighbours.hpp>
#include <engine/spin/interaction/ABC.hpp>

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
        scalarfield exchange_shell_magnitudes{};
        pairfield exchange_pairs{};
        scalarfield exchange_magnitudes{};
    };

    struct Cache
    {
        pairfield exchange_pairs{};
        scalarfield exchange_magnitudes{};
    };

    static bool is_contributing( const Data &, const Cache & cache )
    {
        return !cache.exchange_pairs.empty();
    }

    struct IndexType
    {
        int ispin, jspin, ipair;
    };

    using Index = std::vector<IndexType>;

    static void clearIndex( Index & index )
    {
        index.clear();
    };

    using Energy   = Local::Energy_Functor<Exchange>;
    using Gradient = Local::Gradient_Functor<Exchange>;
    using Hessian  = Local::Hessian_Functor<Exchange>;

    static std::size_t Sparse_Hessian_Size_per_Cell( const Data &, const Cache & cache )
    {
        return cache.exchange_pairs.size() * 2;
    };

    // Calculate the total energy for a single spin to be used in Monte Carlo.
    //      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
    using Energy_Single_Spin = Local::Energy_Single_Spin_Functor<Energy, 2>;

    // Interaction name as string
    static constexpr std::string_view name = "Exchange";

    // we only read from a common source, so multithreaded solutions shouldn't need redundant neighbours.
    static constexpr bool use_redundant_neighbours = false;

    template<typename IndexVector>
    static void applyGeometry(
        const ::Data::Geometry & geometry, const intfield & boundary_conditions, const Data & data, Cache & cache,
        IndexVector & indices )
    {
        using Indexing::idx_from_pair;

        cache.exchange_pairs      = pairfield( 0 );
        cache.exchange_magnitudes = scalarfield( 0 );
        if( !data.exchange_shell_magnitudes.empty() )
        {
            // Generate Exchange neighbours
            intfield exchange_shells( 0 );
            Neighbours::Get_Neighbours_in_Shells(
                geometry, data.exchange_shell_magnitudes.size(), cache.exchange_pairs, exchange_shells,
                use_redundant_neighbours );
            cache.exchange_magnitudes.reserve( cache.exchange_pairs.size() );
            for( std::size_t ipair = 0; ipair < cache.exchange_pairs.size(); ++ipair )
            {
                cache.exchange_magnitudes.push_back( data.exchange_shell_magnitudes[exchange_shells[ipair]] );
            }
        }
        else
        {
            // Use direct list of pairs
            cache.exchange_pairs      = data.exchange_pairs;
            cache.exchange_magnitudes = data.exchange_magnitudes;
            if constexpr( use_redundant_neighbours )
            {
                for( std::size_t i = 0; i < data.exchange_pairs.size(); ++i )
                {
                    const auto & p = data.exchange_pairs[i];
                    const auto & t = p.translations;
                    cache.exchange_pairs.emplace_back( Pair{ p.j, p.i, { -t[0], -t[1], -t[2] } } );
                    cache.exchange_magnitudes.push_back( data.exchange_magnitudes[i] );
                }
            }
        }

#pragma omp parallel for
        for( unsigned int icell = 0; icell < geometry.n_cells_total; ++icell )
        {
            for( unsigned int i_pair = 0; i_pair < cache.exchange_pairs.size(); ++i_pair )
            {
                int ispin = cache.exchange_pairs[i_pair].i + icell * geometry.n_cell_atoms;
                int jspin = idx_from_pair(
                    ispin, boundary_conditions, geometry.n_cells, geometry.n_cell_atoms, geometry.atom_types,
                    cache.exchange_pairs[i_pair] );
                if( jspin >= 0 )
                {
                    std::get<Index>( indices[ispin] ).emplace_back( IndexType{ ispin, jspin, (int)i_pair } );
                    std::get<Index>( indices[jspin] ).emplace_back( IndexType{ ispin, jspin, (int)i_pair } );
                }
            }
        }
    };
};

template<>
template<typename F>
void Exchange::Hessian::operator()( const Index & index, const vectorfield &, F & f ) const
{

    std::for_each(
        begin( index ), end( index ),
        [this, &index, &f]( const auto & idx )
        {
            const int i       = 3 * idx.ispin;
            const int j       = 3 * idx.jspin;
            const auto & i_pair = idx.ipair;

#pragma unroll
            for( int alpha = 0; alpha < 3; ++alpha )
            {
                f( i + alpha, j + alpha, -cache.exchange_magnitudes[i_pair] );
            }
        } );
}

} // namespace Interaction

} // namespace Spin

} // namespace Engine

static_assert(std::is_nothrow_default_constructible<Engine::Spin::Interaction::Exchange::Data>::value);

#endif
