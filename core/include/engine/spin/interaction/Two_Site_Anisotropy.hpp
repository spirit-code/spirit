#pragma once
#ifndef SPIRIT_CORE_ENGINE_SPIN_INTERACTION_TWO_SITE_ANISOTROPY_HPP
#define SPIRIT_CORE_ENGINE_SPIN_INTERACTION_TWO_SITE_ANISOTROPY_HPP

#include <engine/Index_Container.hpp>
#include <engine/Indexing.hpp>
#include <engine/Neighbours.hpp>
#include <engine/Span.hpp>
#include <engine/spin/StateType.hpp>
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
        Data() = default;
        pairfield pairs;
        scalarfield coefficients;
    };

    static bool valid_data( const Data & data )
    {
        return 6 * data.pairs.size() == data.coefficients.size();
    };

    struct Cache
    {
        Cache() = default;
        field<Matrix3> matrices;
    };

    static bool is_contributing( const Data & data, const Cache & )
    {
        return !data.pairs.empty();
    }

    struct Index
    {
        int ispin = 0, jspin = 0, ipair = 0;
    };

    using IndexContainer = Engine::IndexContainer<Two_Site_Anisotropy>;

    // These are the default implementations that hook into
    using Energy   = Functor::Local::Energy_Functor<Functor::Local::DataRef<Two_Site_Anisotropy>>;
    using Gradient = Functor::Local::Gradient_Functor<Functor::Local::DataRef<Two_Site_Anisotropy>>;
    using Hessian  = Functor::Local::Hessian_Functor<Functor::Local::DataRef<Two_Site_Anisotropy>>;

    static std::size_t Sparse_Hessian_Size_per_Cell( const Data &, const Cache & cache )
    {
        return cache.matrices.size() * 9;
    };

    // Calculate the total energy for a single spin to be used in Monte Carlo.
    //      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
    using Energy_Single_Spin = Functor::Local::Energy_Single_Spin_Functor<Energy, 2>;

    // Interaction name as string
    static constexpr std::string_view name = "Two-Site Anisotropy";

    static void
    applyGeometry( const ::Data::Geometry & geometry, const Data & data, Cache & cache, IndexContainer & container )
    {
        using Indexing::idx_from_pair;
        // convert (eigen vector, eigen value) pairs to symmetrical matrices
        cache.matrices = field<Matrix3>( data.pairs.size(), Matrix3::Zero() );
        for( unsigned int imatrix = 0; imatrix < cache.matrices.size(); ++imatrix )
        {
            const scalar Kxx = data.coefficients[6 * imatrix + 0];
            const scalar Kyy = data.coefficients[6 * imatrix + 1];
            const scalar Kzz = data.coefficients[6 * imatrix + 2];
            const scalar Kyz = data.coefficients[6 * imatrix + 3];
            const scalar Kxz = data.coefficients[6 * imatrix + 4];
            const scalar Kxy = data.coefficients[6 * imatrix + 5];
            cache.matrices[imatrix] << Kxx, Kxy, Kxz, Kxy, Kyy, Kyz, Kxz, Kyz, Kzz;
        }

        // calculate local indices based off provided pairs
        auto indices = std::vector( geometry.nos, field<Index>{} );
        for( int icell = 0; icell < geometry.n_cells_total; ++icell )
        {
            for( int ipair = 0; ipair < data.pairs.size(); ++ipair )
            {
                int ispin = data.pairs[ipair].i + icell * geometry.n_cell_atoms;
                int jspin = Indexing::idx_from_pair(
                    ispin, geometry.boundary_conditions, geometry.n_cells, geometry.n_cell_atoms, geometry.atom_types,
                    data.pairs[ipair] );
                if( jspin >= 0 )
                {
                    indices[ispin].push_back( Index{ ispin, jspin, ipair } );
                    indices[jspin].push_back( Index{ jspin, ispin, ipair } );
                }
            }
        }

        container = make_index_container<Two_Site_Anisotropy>( std::move( indices ) );
    };
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
inline scalar Two_Site_Anisotropy::Energy::operator()( Span<const Index> index, quantity<const Vector3 *> state ) const
{
    scalar energy = 0.0;
    for( const Index & idx : index )
    {
        energy -= state.spin[idx.ispin].dot( matrices[idx.ipair] * state.spin[idx.jspin] );
    }
    return 0.5 * energy;
}

template<>
inline Vector3
Two_Site_Anisotropy::Gradient::operator()( Span<const Index> index, quantity<const Vector3 *> state ) const
{
    Vector3 gradient = Vector3::Zero();
    for( const Index & idx : index )
    {
        gradient -= matrices[idx.ipair] * state.spin[idx.jspin];
    }
    return gradient;
}

template<>
template<typename Callable>
void Two_Site_Anisotropy::Hessian::operator()( Span<const Index> index, const StateType &, Callable & hessian ) const
{
    for( const Index & idx : index )
    {
        const int i      = 3 * idx.ispin;
        const int j      = 3 * idx.jspin;
        const auto ipair = idx.ipair;

        for( int alpha = 0; alpha < 3; ++alpha )
        {
            for( int beta = 0; beta < 3; ++beta )
            {
                hessian( i + alpha, j + beta, -matrices[ipair]( alpha, beta ) );
            }
        }
    }
}

} // namespace Interaction

} // namespace Spin

} // namespace Engine

#endif
