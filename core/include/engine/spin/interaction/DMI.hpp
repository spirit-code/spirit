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
        pairfield pairs{};
        scalarfield magnitudes{};
        vectorfield normals{};

        scalarfield shell_magnitudes{};
        int shell_chirality = 0;
    };

    static bool valid_data( const Data & data )
    {
        if( !data.shell_magnitudes.empty() && data.shell_chirality != SPIRIT_CHIRALITY_NEEL
            && data.shell_chirality != SPIRIT_CHIRALITY_BLOCH && data.shell_chirality != SPIRIT_CHIRALITY_NEEL_INVERSE
            && data.shell_chirality != SPIRIT_CHIRALITY_BLOCH_INVERSE )
            return false;
        if( !data.pairs.empty()
            && ( data.pairs.size() != data.magnitudes.size() || data.pairs.size() != data.normals.size() ) )
            return false;

        return true;
    };

    struct Cache
    {
        pairfield pairs{};
        scalarfield magnitudes{};
        vectorfield normals{};
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

    using Energy   = Local::Energy_Functor<DMI>;
    using Gradient = Local::Gradient_Functor<DMI>;
    using Hessian  = Local::Hessian_Functor<DMI>;

    static std::size_t Sparse_Hessian_Size_per_Cell( const Data &, const Cache & cache )
    {
        return cache.pairs.size() * 3;
    };

    // Calculate the total energy for a single spin to be used in Monte Carlo.
    //      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
    using Energy_Single_Spin = Local::Energy_Single_Spin_Functor<Energy, 2>;

    // Interaction name as string
    static constexpr std::string_view name = "DMI";

    // we have to use redundant neighbours here, because the 'inverse' pair has opposite sign
    // TODO: check if introducing a sign factor is faster
    static constexpr bool use_redundant_neighbours = true;

    template<typename IndexVector>
    static void applyGeometry(
        const ::Data::Geometry & geometry, const intfield & boundary_conditions, const Data & data, Cache & cache,
        IndexVector & indices )
    {
        using Indexing::idx_from_pair;

        cache.pairs      = pairfield( 0 );
        cache.magnitudes = scalarfield( 0 );
        cache.normals    = vectorfield( 0 );
        if( !data.shell_magnitudes.empty() )
        {
            // Generate DMI neighbours and normals
            intfield dmi_shells( 0 );
            Neighbours::Get_Neighbours_in_Shells(
                geometry, data.shell_magnitudes.size(), cache.pairs, dmi_shells, use_redundant_neighbours );
            for( std::size_t ineigh = 0; ineigh < cache.pairs.size(); ++ineigh )
            {
                cache.normals.push_back(
                    Neighbours::DMI_Normal_from_Pair( geometry, cache.pairs[ineigh], data.shell_chirality ) );
                cache.magnitudes.push_back( data.shell_magnitudes[dmi_shells[ineigh]] );
            }
        }
        else
        {
            // Use direct list of pairs
            cache.pairs      = data.pairs;
            cache.magnitudes = data.magnitudes;
            cache.normals    = data.normals;
            if( use_redundant_neighbours )
            {
                for( std::size_t i = 0; i < data.pairs.size(); ++i )
                {
                    const auto & p = data.pairs[i];
                    const auto & t = p.translations;
                    cache.pairs.emplace_back( Pair{ p.j, p.i, { -t[0], -t[1], -t[2] } } );
                    cache.magnitudes.emplace_back( data.magnitudes[i] );
                    cache.normals.emplace_back( -data.normals[i] );
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
        [this, &index, &f]( const DMI::IndexType & idx )
        {
            const int i       = 3 * idx.ispin;
            const int j       = 3 * idx.jspin;
            const auto i_pair = idx.ipair;

            f( i + 2, j + 1, cache.magnitudes[i_pair] * cache.normals[i_pair][0] );
            f( i + 1, j + 2, -cache.magnitudes[i_pair] * cache.normals[i_pair][0] );
            f( i + 0, j + 2, cache.magnitudes[i_pair] * cache.normals[i_pair][1] );
            f( i + 2, j + 0, -cache.magnitudes[i_pair] * cache.normals[i_pair][1] );
            f( i + 1, j + 0, cache.magnitudes[i_pair] * cache.normals[i_pair][2] );
            f( i + 0, j + 1, -cache.magnitudes[i_pair] * cache.normals[i_pair][2] );
            if constexpr( !DMI::use_redundant_neighbours )
            {
                f( j + 1, i + 2, cache.magnitudes[i_pair] * cache.normals[i_pair][0] );
                f( j + 2, i + 1, -cache.magnitudes[i_pair] * cache.normals[i_pair][0] );
                f( j + 2, i + 0, cache.magnitudes[i_pair] * cache.normals[i_pair][1] );
                f( j + 0, i + 2, -cache.magnitudes[i_pair] * cache.normals[i_pair][1] );
                f( j + 0, i + 1, cache.magnitudes[i_pair] * cache.normals[i_pair][2] );
                f( j + 1, i + 0, -cache.magnitudes[i_pair] * cache.normals[i_pair][2] );
            }

            // #if !( defined( SPIRIT_USE_OPENMP ) || defined( SPIRIT_USE_CUDA ) )
            // #endif
        } );
};

} // namespace Interaction

} // namespace Spin

} // namespace Engine
#endif
