#pragma once
#ifndef SPIRIT_CORE_ENGINE_INTERACTION_TWO_ION_ANISOTROPY_HPP
#define SPIRIT_CORE_ENGINE_INTERACTION_TWO_ION_ANISOTROPY_HPP

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

struct Two_Site_Anisotropy
{
    using state_t = StateType;

    struct Data
    {
        pairfield pairs{};
        matrixfield matrices{};

        Data() = default;
        Data( pairfield pairs, matrixfield matrices )
                : pairs( std::move( pairs ) ), matrices( std::move( matrices ) ) {};
    };

    static bool valid_data( const Data & data )
    {
        if( !data.pairs.empty() && ( data.pairs.size() != data.matrices.size() ) )
            return false;

        return true;
    };

    struct Cache
    {
        pairfield pairs{};
        matrixfield matrices{};
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

    using Energy   = Functor::Local::Energy_Functor<Functor::Local::DataRef<Two_Site_Anisotropy>>;
    using Gradient = Functor::Local::Gradient_Functor<Functor::Local::DataRef<Two_Site_Anisotropy>>;
    using Hessian  = Functor::Local::Hessian_Functor<Functor::Local::DataRef<Two_Site_Anisotropy>>;

    static std::size_t Sparse_Hessian_Size_per_Cell( const Data &, const Cache & cache )
    {
        return cache.pairs.size() * 9;
    };

    // Calculate the total energy for a single spin to be used in Monte Carlo.
    //      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
    using Energy_Single_Spin = Functor::Local::Energy_Single_Spin_Functor<Energy, 2>;

    // Interaction name as string
    static constexpr std::string_view name = "Two Site Anisotropy";

    template<typename IndexStorageVector>
    static void applyGeometry(
        const ::Data::Geometry & geometry, const intfield & boundary_conditions, const Data & data, Cache & cache,
        IndexStorageVector & indices )
    {
        using Indexing::idx_from_pair;

        // redundant neighbours are captured when expanding pairs below
        static constexpr bool use_redundant_neighbours = false;

        // Use direct list of pairs
        // TODO: find a more convenient approach for implementing these in terms of neighbours.
        cache.pairs    = data.pairs;
        cache.matrices = data.matrices;

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
            };
        }
    }
};

template<>
struct Functor::Local::DataRef<Two_Site_Anisotropy>
{
    using Interaction = Two_Site_Anisotropy;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;

    DataRef( const Data & data, const Cache & cache ) noexcept
            : is_contributing( Interaction::is_contributing( data, cache ) ), matrices( cache.matrices.data() )
    {
    }

    const bool is_contributing;

protected:
    const Matrix3 * matrices;
};

template<>
inline scalar Two_Site_Anisotropy::Energy::operator()( const Index & index, quantity<const Vector3 *> state ) const
{
    // don't need to check for `is_contributing` here, because `transform_reduce` will short circuit properly
    return Backend::transform_reduce(
        index.begin(), index.end(), scalar( 0.0 ), Backend::plus<scalar>{},
        [this, state] SPIRIT_LAMBDA( const Two_Site_Anisotropy::IndexType & idx ) -> scalar
        { return state.spin[idx.ispin].dot( matrices[idx.ipair] * state.spin[idx.jspin] ); } );
}

template<>
inline Vector3 Two_Site_Anisotropy::Gradient::operator()( const Index & index, quantity<const Vector3 *> state ) const
{
    // don't need to check for `is_contributing` here, because `transform_reduce` will short circuit properly
    return Backend::transform_reduce(
        index.begin(), index.end(), Vector3{ 0.0, 0.0, 0.0 }, Backend::plus<Vector3>{},
        [this, state] SPIRIT_LAMBDA( const Two_Site_Anisotropy::IndexType & idx ) -> Vector3
        { return matrices[idx.ipair] * state.spin[idx.jspin]; } );
}

template<>
template<typename Callable>
void Two_Site_Anisotropy::Hessian::operator()( const Index & index, const StateType &, Callable & hessian ) const
{
    // don't need to check for `is_contributing` here, because `for_each` will short circuit properly
    Backend::cpu::for_each(
        index.begin(), index.end(),
        [this, &index, &hessian]( const Two_Site_Anisotropy::IndexType & idx )
        {
            const int i = 3 * idx.ispin;
            const int j = 3 * idx.jspin;

            for( int alpha = 0; alpha < 3; ++alpha )
            {
                for( int beta = 0; beta < 3; ++beta )
                {
                    hessian( i + alpha, j + beta, matrices[idx.ipair]( alpha, beta ) );
                }
            }
        } );
};

} // namespace Interaction

} // namespace Spin

} // namespace Engine
#endif
