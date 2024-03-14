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
        pairfield pairs{};
        scalarfield magnitudes{};

        scalarfield shell_magnitudes{};

        Data() = default;
        Data( pairfield pairs, scalarfield magnitudes )
                : pairs( std::move( pairs ) ), magnitudes( std::move( magnitudes ) ){};

        Data( scalarfield shell_magnitudes ) : shell_magnitudes( std::move( shell_magnitudes ) ){};
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
        return cache.pairs.size() * 2;
    };

    // Calculate the total energy for a single spin to be used in Monte Carlo.
    //      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
    using Energy_Single_Spin = Local::Energy_Single_Spin_Functor<Energy, 2>;

    // Interaction name as string
    static constexpr std::string_view name = "Exchange";

    // the interaction is symmetrical with respect to swapping atoms in a pair,
    // so multithreaded solutions shouldn't need redundant neighbours.
    static constexpr bool use_redundant_neighbours = false;

    template<typename IndexVector>
    static void applyGeometry(
        const ::Data::Geometry & geometry, const intfield & boundary_conditions, const Data & data, Cache & cache,
        IndexVector & indices )
    {
        using Indexing::idx_from_pair;

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
            if constexpr( use_redundant_neighbours )
            {
                for( std::size_t i = 0; i < data.pairs.size(); ++i )
                {
                    const auto & p = data.pairs[i];
                    const auto & t = p.translations;
                    cache.pairs.emplace_back( Pair{ p.j, p.i, { -t[0], -t[1], -t[2] } } );
                    cache.magnitudes.push_back( data.magnitudes[i] );
                }
            }
        }

#pragma omp parallel for
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
                    std::get<Index>( indices[ispin] ).emplace_back( IndexType{ ispin, jspin, (int)i_pair } );
                    std::get<Index>( indices[jspin] ).emplace_back( IndexType{ jspin, ispin, (int)i_pair } );
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
            const int i         = 3 * idx.ispin;
            const int j         = 3 * idx.jspin;
            const auto & i_pair = idx.ipair;

#pragma unroll
            for( int alpha = 0; alpha < 3; ++alpha )
            {
                f( i + alpha, j + alpha, -cache.magnitudes[i_pair] );
            }
        } );
}

} // namespace Interaction

} // namespace Spin

} // namespace Engine

#endif
