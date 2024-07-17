#pragma once
#ifndef SPIRIT_CORE_ENGINE_INTERACTION_ANISOTROPY_HPP
#define SPIRIT_CORE_ENGINE_INTERACTION_ANISOTROPY_HPP

#include <engine/Indexing.hpp>
#include <engine/spin/interaction/Functor_Prototypes.hpp>

#include <Eigen/Dense>

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
        intfield indices{};
        scalarfield magnitudes{};
        vectorfield normals{};

        Data() = default;
        Data( intfield indices, scalarfield magnitudes, vectorfield normals )
                : indices( std::move( indices ) ),
                  magnitudes( std::move( magnitudes ) ),
                  normals( std::move( normals ) ) {};
    };

    static bool valid_data( const Data & data )
    {
        using std::begin, std::end;

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

    using Index        = const IndexType *;
    using IndexStorage = Backend::optional<IndexType>;

    using Energy   = Functor::Local::Energy_Functor<Functor::Local::DataRef<Anisotropy>>;
    using Gradient = Functor::Local::Gradient_Functor<Functor::Local::DataRef<Anisotropy>>;
    using Hessian  = Functor::Local::Hessian_Functor<Functor::Local::DataRef<Anisotropy>>;

    static std::size_t Sparse_Hessian_Size_per_Cell( const Data & data, const Cache & )
    {
        return data.indices.size() * 9;
    };

    // Calculate the total energy for a single spin to be used in Monte Carlo.
    //      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
    using Energy_Single_Spin = Functor::Local::Energy_Single_Spin_Functor<Energy, 1>;

    // Interaction name as string
    static constexpr std::string_view name = "Anisotropy";

    template<typename IndexStorageVector>
    static void applyGeometry(
        const ::Data::Geometry & geometry, const intfield &, const Data & data, Cache &, IndexStorageVector & indices )
    {
        using Indexing::check_atom_type;

        for( int icell = 0; icell < geometry.n_cells_total; ++icell )
        {
            for( int iani = 0; iani < data.indices.size(); ++iani )
            {
                int ispin = icell * geometry.n_cell_atoms + data.indices[iani];
                if( check_atom_type( geometry.atom_types[ispin] ) )
                    Backend::get<IndexStorage>( indices[ispin] ) = IndexType{ ispin, iani };
            }
        }
    };
};

template<>
struct Functor::Local::DataRef<Anisotropy>
{
    using Interaction = Anisotropy;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;

    DataRef( const Data & data, const Cache & cache ) noexcept
            : is_contributing( Interaction::is_contributing( data, cache ) ),
              normals( data.normals.data() ),
              magnitudes( data.magnitudes.data() ) {};

    const bool is_contributing;

protected:
    const Vector3 * normals;
    const scalar * magnitudes;
};

template<>
inline scalar Anisotropy::Energy::operator()( const Index & index, const Vector3 * spins ) const
{
    if( is_contributing && index != nullptr )
    {
        const auto & [ispin, iani] = *index;
        const auto d               = normals[iani].dot( spins[ispin] );
        return -magnitudes[iani] * d * d;
    }
    else
    {
        return 0;
    }
}

template<>
inline Vector3 Anisotropy::Gradient::operator()( const Index & index, const Vector3 * spins ) const
{
    if( is_contributing && index != nullptr )
    {
        const auto & [ispin, iani] = *index;
        return -2.0 * magnitudes[iani] * normals[iani] * normals[iani].dot( spins[ispin] );
    }
    else
    {
        return Vector3::Zero();
    }
}

template<>
template<typename Callable>
void Anisotropy::Hessian::operator()( const Index & index, const vectorfield &, Callable & hessian ) const
{
    if( !is_contributing || index == nullptr )
        return;

    const auto & [ispin, iani] = *index;

#pragma unroll
    for( int alpha = 0; alpha < 3; ++alpha )
    {
#pragma unroll
        for( int beta = 0; beta < 3; ++beta )
        {
            const int i = 3 * ispin + alpha;
            const int j = 3 * ispin + alpha;
            hessian( i, j, -2.0 * magnitudes[iani] * normals[iani][alpha] * normals[iani][beta] );
        }
    }
}

} // namespace Interaction

} // namespace Spin

} // namespace Engine
#endif
