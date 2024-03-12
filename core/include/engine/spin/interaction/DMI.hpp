#pragma once
#ifndef SPIRIT_CORE_ENGINE_INTERACTION_DMI_HPP
#define SPIRIT_CORE_ENGINE_INTERACTION_DMI_HPP

#include <engine/Indexing.hpp>
#include <engine/Neighbours.hpp>
#include <engine/spin/interaction/ABC.hpp>

namespace Engine
{

namespace Spin
{

namespace Interaction
{

struct DMI
{
    using state_t = vectorfield;

    struct Data
    {
        pairfield dmi_pairs{};
        scalarfield dmi_magnitudes{};
        vectorfield dmi_normals{};

        scalarfield dmi_shell_magnitudes{};
        int dmi_shell_chirality = 0;
    };

    struct Cache
    {
        pairfield dmi_pairs{};
        scalarfield dmi_magnitudes{};
        vectorfield dmi_normals{};
    };

    static bool is_contributing( const Data &, const Cache & cache )
    {
        return !cache.dmi_pairs.empty();
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

    using Energy   = Local::Energy_Functor<DMI>;
    using Gradient = Local::Gradient_Functor<DMI>;
    using Hessian  = Local::Hessian_Functor<DMI>;

    static std::size_t Sparse_Hessian_Size_per_Cell( const Data &, const Cache & cache )
    {
        return cache.dmi_pairs.size() * 3;
    };

    // Calculate the total energy for a single spin to be used in Monte Carlo.
    //      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
    using Energy_Single_Spin = Local::Energy_Single_Spin_Functor<Energy, 2>;

    // Interaction name as string
    static constexpr std::string_view name = "DMI";

    // we only read from a common source, so multithreaded solutions shouldn't need redundant neighbours.
    static constexpr bool use_redundant_neighbours = false;

    template<typename IndexVector>
    static void applyGeometry(
        const ::Data::Geometry & geometry, const intfield & boundary_conditions, const Data & data, Cache & cache,
        IndexVector & indices )
    {
        using Indexing::idx_from_pair;

        cache.dmi_pairs      = pairfield( 0 );
        cache.dmi_magnitudes = scalarfield( 0 );
        cache.dmi_normals    = vectorfield( 0 );
        if( !data.dmi_shell_magnitudes.empty() )
        {
            // Generate DMI neighbours and normals
            intfield dmi_shells( 0 );
            Neighbours::Get_Neighbours_in_Shells(
                geometry, data.dmi_shell_magnitudes.size(), cache.dmi_pairs, dmi_shells, use_redundant_neighbours );
            for( std::size_t ineigh = 0; ineigh < cache.dmi_pairs.size(); ++ineigh )
            {
                cache.dmi_normals.push_back(
                    Neighbours::DMI_Normal_from_Pair( geometry, cache.dmi_pairs[ineigh], data.dmi_shell_chirality ) );
                cache.dmi_magnitudes.push_back( data.dmi_shell_magnitudes[dmi_shells[ineigh]] );
            }
        }
        else
        {
            // Use direct list of pairs
            cache.dmi_pairs      = data.dmi_pairs;
            cache.dmi_magnitudes = data.dmi_magnitudes;
            cache.dmi_normals    = data.dmi_normals;
            if( use_redundant_neighbours )
            {
                for( std::size_t i = 0; i < data.dmi_pairs.size(); ++i )
                {
                    const auto & p = data.dmi_pairs[i];
                    const auto & t = p.translations;
                    cache.dmi_pairs.emplace_back( Pair{ p.j, p.i, { -t[0], -t[1], -t[2] } } );
                    cache.dmi_magnitudes.emplace_back( data.dmi_magnitudes[i] );
                    cache.dmi_normals.emplace_back( -data.dmi_normals[i] );
                }
            }
        }

#pragma omp parallel for
        for( unsigned int icell = 0; icell < geometry.n_cells_total; ++icell )
        {
            for( unsigned int i_pair = 0; i_pair < cache.dmi_pairs.size(); ++i_pair )
            {
                int ispin = cache.dmi_pairs[i_pair].i + icell * geometry.n_cell_atoms;
                int jspin = idx_from_pair(
                    ispin, boundary_conditions, geometry.n_cells, geometry.n_cell_atoms, geometry.atom_types,
                    cache.dmi_pairs[i_pair] );
                if( jspin >= 0 )
                {
                    std::get<Index>( indices[ispin] ).emplace_back( IndexType{ ispin, jspin, (int)i_pair } );
                    std::get<Index>( indices[jspin] ).emplace_back( IndexType{ ispin, jspin, (int)i_pair } );
                }
            };
        }
    }
};

template<>
template<typename F>
void DMI::Hessian::operator()( const Index & index, const vectorfield &, F & f ) const
{

    std::for_each(
        begin( index ), end( index ),
        [this, &index, &f]( const auto & idx )
        {
            const int i       = 3 * idx.ispin;
            const int j       = 3 * idx.jspin;
            const auto i_pair = idx.ipair;

            f( i + 2, j + 1, cache.dmi_magnitudes[i_pair] * cache.dmi_normals[i_pair][0] );
            f( i + 1, j + 2, -cache.dmi_magnitudes[i_pair] * cache.dmi_normals[i_pair][0] );
            f( i + 0, j + 2, cache.dmi_magnitudes[i_pair] * cache.dmi_normals[i_pair][1] );
            f( i + 2, j + 0, -cache.dmi_magnitudes[i_pair] * cache.dmi_normals[i_pair][1] );
            f( i + 1, j + 0, cache.dmi_magnitudes[i_pair] * cache.dmi_normals[i_pair][2] );
            f( i + 0, j + 1, -cache.dmi_magnitudes[i_pair] * cache.dmi_normals[i_pair][2] );
            if constexpr( !DMI::use_redundant_neighbours )
            {
                f( j + 1, i + 2, cache.dmi_magnitudes[i_pair] * cache.dmi_normals[i_pair][0] );
                f( j + 2, i + 1, -cache.dmi_magnitudes[i_pair] * cache.dmi_normals[i_pair][0] );
                f( j + 2, i + 0, cache.dmi_magnitudes[i_pair] * cache.dmi_normals[i_pair][1] );
                f( j + 0, i + 2, -cache.dmi_magnitudes[i_pair] * cache.dmi_normals[i_pair][1] );
                f( j + 0, i + 1, cache.dmi_magnitudes[i_pair] * cache.dmi_normals[i_pair][2] );
                f( j + 1, i + 0, -cache.dmi_magnitudes[i_pair] * cache.dmi_normals[i_pair][2] );
            }

            // #if !( defined( SPIRIT_USE_OPENMP ) || defined( SPIRIT_USE_CUDA ) )
            // #endif
        } );
};

} // namespace Interaction

} // namespace Spin

} // namespace Engine
#endif
