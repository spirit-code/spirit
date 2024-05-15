#pragma once
#ifndef SPIRIT_CORE_ENGINE_INTERACTION_CUBIC_ANISOTROPY_HPP
#define SPIRIT_CORE_ENGINE_INTERACTION_CUBIC_ANISOTROPY_HPP

#include <engine/Indexing.hpp>
#include <engine/spin/interaction/Functor_Prototypes.hpp>
#include <utility/Fastpow.hpp>

#include <Eigen/Dense>

namespace Engine
{

namespace Spin
{

namespace Interaction
{

struct Cubic_Anisotropy
{
    using state_t = vectorfield;

    struct Data
    {
        intfield indices;
        scalarfield magnitudes;

        Data( intfield indices, scalarfield magnitudes )
                : indices( std::move( indices ) ), magnitudes( std::move( magnitudes ) ){};
    };

    static bool valid_data( const Data & data )
    {
        return data.magnitudes.size() == data.indices.size();
    };

    struct Cache
    {
    };

    static bool is_contributing( const Data & data, const Cache & )
    {
        return !data.indices.empty();
    }

    struct IndexType
    {
        int ispin, iani;
    };

    using Index        = const IndexType *;
    using IndexStorage = Backend::optional<IndexType>;

    using Energy   = Functor::Local::Energy_Functor<Functor::Local::DataRef<Cubic_Anisotropy>>;
    using Gradient = Functor::Local::Gradient_Functor<Functor::Local::DataRef<Cubic_Anisotropy>>;
    using Hessian  = Functor::Local::Hessian_Functor<Functor::Local::DataRef<Cubic_Anisotropy>>;

    static std::size_t Sparse_Hessian_Size_per_Cell( const Data &, const Cache & )
    {
        return 0;
    };

    // Calculate the total energy for a single spin to be used in Monte Carlo.
    //      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
    using Energy_Single_Spin = Functor::Local::Energy_Single_Spin_Functor<Energy, 1>;

    // Interaction name as string
    static constexpr std::string_view name = "Cubic Anisotropy";

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
struct Functor::Local::DataRef<Cubic_Anisotropy>
{
    using Interaction = Cubic_Anisotropy;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;

    DataRef( const Data & data, const Cache & cache ) noexcept
            : is_contributing( Interaction::is_contributing( data, cache ) ),
              magnitudes( data.magnitudes.data() )
    {
    }

    const bool is_contributing;

protected:
    const scalar * magnitudes;
};

template<>
inline scalar Cubic_Anisotropy::Energy::operator()( const Index & index, const Vector3 * spins ) const
{
    using Utility::fastpow;
    scalar result = 0;
    if( !is_contributing || index == nullptr )
        return result;

    const auto & [ispin, iani] = *index;
    return -0.5 * magnitudes[iani]
           * ( fastpow( spins[ispin][0], 4u ) + fastpow( spins[ispin][1], 4u ) + fastpow( spins[ispin][2], 4u ) );
}

template<>
inline Vector3 Cubic_Anisotropy::Gradient::operator()( const Index & index, const Vector3 * spins ) const
{
    using Utility::fastpow;
    Vector3 result = Vector3::Zero();
    if( !is_contributing || index == nullptr )
        return result;

    const auto & [ispin, iani] = *index;

    for( int icomp = 0; icomp < 3; ++icomp )
    {
        result[icomp] = -2.0 * magnitudes[iani] * fastpow( spins[ispin][icomp], 3u );
    }
    return result;
}

template<>
template<typename Callable>
void Cubic_Anisotropy::Hessian::operator()( const Index & index, const vectorfield & spins, Callable & hessian ) const
{
    // TODO: Not yet implemented
}

} // namespace Interaction

} // namespace Spin

} // namespace Engine
#endif
