#pragma once
#ifndef SPIRIT_CORE_ENGINE_INTERACTION_DMI_HPP
#define SPIRIT_CORE_ENGINE_INTERACTION_DMI_HPP

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

        Data() = default;
        Data( pairfield pairs, scalarfield magnitudes, vectorfield normals )
                : pairs( std::move( pairs ) ), magnitudes( std::move( magnitudes ) ), normals( std::move( normals ) ){};

        Data( scalarfield shell_magnitudes, int shell_chirality )
                : shell_magnitudes( std::move( shell_magnitudes ) ), shell_chirality( shell_chirality ){};
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
        bool inverse;
    };

    using Index        = Engine::Span<const IndexType>;
    using IndexStorage = Backend::vector<IndexType>;

    using Energy   = Functor::Local::Energy_Functor<Functor::Local::DataRef<DMI>>;
    using Gradient = Functor::Local::Gradient_Functor<Functor::Local::DataRef<DMI>>;
    using Hessian  = Functor::Local::Hessian_Functor<Functor::Local::DataRef<DMI>>;

    static std::size_t Sparse_Hessian_Size_per_Cell( const Data &, const Cache & cache )
    {
        return cache.pairs.size() * 3;
    };

    // Calculate the total energy for a single spin to be used in Monte Carlo.
    //      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
    using Energy_Single_Spin = Functor::Local::Energy_Single_Spin_Functor<Energy, 2>;

    // Interaction name as string
    static constexpr std::string_view name = "DMI";

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
                    Backend::get<IndexStorage>( indices[ispin] )
                        .push_back( IndexType{ ispin, jspin, (int)i_pair, false } );
                    Backend::get<IndexStorage>( indices[jspin] )
                        .push_back( IndexType{ jspin, ispin, (int)i_pair, true } );
                }
            };
        }
    }
};

template<>
struct Functor::Local::DataRef<DMI>
{
    using Interaction = DMI;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;

    DataRef( const Data & data, const Cache & cache ) noexcept
            : is_contributing( Interaction::is_contributing( data, cache ) ),
              magnitudes( cache.magnitudes.data() ),
              normals( cache.normals.data() )
    {
    }

    const bool is_contributing;

protected:
    const scalar * magnitudes;
    const Vector3 * normals;
};

template<>
inline scalar DMI::Energy::operator()( const Index & index, const Vector3 * spins ) const
{
    // don't need to check for `is_contributing` here, because `transform_reduce` will short circuit properly
    return Backend::transform_reduce(
        index.begin(), index.end(), scalar( 0.0 ), Backend::plus<scalar>{},
        [this, spins] SPIRIT_LAMBDA( const DMI::IndexType & idx ) -> scalar
        {
            const auto & [ispin, jspin, i_pair, inverse] = idx;
            return ( inverse ? 0.5 : -0.5 ) * magnitudes[i_pair]
                   * normals[i_pair].dot( spins[ispin].cross( spins[jspin] ) );
        } );
}

template<>
inline Vector3 DMI::Gradient::operator()( const Index & index, const Vector3 * spins ) const
{
    // don't need to check for `is_contributing` here, because `transform_reduce` will short circuit properly
    return Backend::transform_reduce(
        index.begin(), index.end(), Vector3{ 0.0, 0.0, 0.0 }, Backend::plus<Vector3>{},
        [this, spins] SPIRIT_LAMBDA( const DMI::IndexType & idx ) -> Vector3
        {
            const auto & [ispin, jspin, i_pair, inverse] = idx;
            return ( inverse ? 1.0 : -1.0 ) * magnitudes[i_pair] * spins[jspin].cross( normals[i_pair] );
        } );
}

template<>
template<typename Callable>
void DMI::Hessian::operator()( const Index & index, const vectorfield &, Callable & hessian ) const
{
    // don't need to check for `is_contributing` here, because `for_each` will short circuit properly
    Backend::cpu::for_each(
        index.begin(), index.end(),
        [this, &index, &hessian]( const DMI::IndexType & idx )
        {
            const int i       = 3 * idx.ispin;
            const int j       = 3 * idx.jspin;
            const auto i_pair = idx.ipair;
            const scalar sign = ( idx.inverse ? -1.0 : 1.0 );

            hessian( i + 2, j + 1, sign * magnitudes[i_pair] * normals[i_pair][0] );
            hessian( i + 1, j + 2, -sign * magnitudes[i_pair] * normals[i_pair][0] );
            hessian( i + 0, j + 2, sign * magnitudes[i_pair] * normals[i_pair][1] );
            hessian( i + 2, j + 0, -sign * magnitudes[i_pair] * normals[i_pair][1] );
            hessian( i + 1, j + 0, sign * magnitudes[i_pair] * normals[i_pair][2] );
            hessian( i + 0, j + 1, -sign * magnitudes[i_pair] * normals[i_pair][2] );
        } );
};

} // namespace Interaction

} // namespace Spin

} // namespace Engine
#endif
