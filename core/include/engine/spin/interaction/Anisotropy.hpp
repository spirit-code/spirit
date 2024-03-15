#pragma once
#ifndef SPIRIT_CORE_ENGINE_INTERACTION_ANISOTROPY_HPP
#define SPIRIT_CORE_ENGINE_INTERACTION_ANISOTROPY_HPP

#include <engine/Indexing.hpp>
#include <engine/spin/interaction/Functor_Prototpyes.hpp>

#include <Eigen/Dense>

#include <optional>

namespace Engine
{

namespace Spin
{

namespace Interaction
{

struct Anisotropy
{
    using state_t = vectorfield;

    struct Data
    {
        intfield indices;
        scalarfield magnitudes;
        vectorfield normals;

        Data( intfield indices, scalarfield magnitudes, vectorfield normals )
                : indices( std::move( indices ) ),
                  magnitudes( std::move( magnitudes ) ),
                  normals( std::move( normals ) ){};
    };

    static bool valid_data( const Data & data )
    {
        if( data.indices.size() != data.magnitudes.size() || data.indices.size() != data.normals.size() )
            return false;
        if( std::any_of( begin( data.indices ), end( data.indices ), []( const int & i ) { return i < 0; } ) )
            return false;

        return true;
    }

    struct Cache
    {
    };

    static bool is_contributing( const Data & data, const Cache & )
    {
        return !data.indices.empty();
    };

    struct IndexType
    {
        int ispin, iani;
    };

    using Index = std::optional<IndexType>;

    using Energy   = Functor::Local::Energy_Functor<Anisotropy>;
    using Gradient = Functor::Local::Gradient_Functor<Anisotropy>;
    using Hessian  = Functor::Local::Hessian_Functor<Anisotropy>;

    static std::size_t Sparse_Hessian_Size_per_Cell( const Data & data, const Cache & )
    {
        return data.indices.size() * 9;
    };

    // Calculate the total energy for a single spin to be used in Monte Carlo.
    //      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
    using Energy_Single_Spin = Functor::Local::Energy_Single_Spin_Functor<Energy, 1>;

    // Interaction name as string
    static constexpr std::string_view name = "Anisotropy";

    template<typename IndexVector>
    static void applyGeometry(
        const ::Data::Geometry & geometry, const intfield &, const Data & data, Cache &, IndexVector & indices )
    {
        using Indexing::check_atom_type;

        for( int icell = 0; icell < geometry.n_cells_total; ++icell )
        {
            for( int iani = 0; iani < data.indices.size(); ++iani )
            {
                int ispin = icell * geometry.n_cell_atoms + data.indices[iani];
                if( check_atom_type( geometry.atom_types[ispin] ) )
                    std::get<Index>( indices[ispin] ) = IndexType{ ispin, iani };
            }
        }
    };
};

template<>
inline scalar Anisotropy::Energy::operator()( const Index & index, const vectorfield & spins ) const
{
    if( index.has_value() )
    {
        const auto & [ispin, iani] = *index;
        const auto d               = data.normals[iani].dot( spins[ispin] );
        return -data.magnitudes[iani] * d * d;
    }
    else
    {
        return 0;
    }
}

template<>
inline Vector3 Anisotropy::Gradient::operator()( const Index & index, const vectorfield & spins ) const
{
    if( index.has_value() )
    {
        const auto & [ispin, iani] = *index;
        return -2.0 * data.magnitudes[iani] * data.normals[iani]
               * data.normals[iani].dot( spins[ispin] );
    }
    else
    {
        return Vector3::Zero();
    }
}

template<>
template<typename F>
void Anisotropy::Hessian::operator()( const Index & index, const vectorfield &, F & f ) const
{
    if( !index.has_value() )
        return;

    const auto & [ispin, iani] = *index;

#pragma unroll
    for( int alpha = 0; alpha < 3; ++alpha )
    {
#pragma unroll
        for( int beta = 0; beta < 3; ++beta )
        {
            int i = 3 * ispin + alpha;
            int j = 3 * ispin + alpha;
            f( i, j, -2.0 * data.magnitudes[iani] * data.normals[iani][alpha] * data.normals[iani][beta] );
        }
    }
}

} // namespace Interaction

} // namespace Spin

} // namespace Engine
#endif
