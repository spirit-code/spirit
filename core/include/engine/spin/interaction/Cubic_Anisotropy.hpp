#pragma once
#ifndef SPIRIT_CORE_ENGINE_INTERACTION_CUBIC_ANISOTROPY_HPP
#define SPIRIT_CORE_ENGINE_INTERACTION_CUBIC_ANISOTROPY_HPP

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

    using Index = std::optional<IndexType>;

    static void clearIndex( Index & index )
    {
        index.reset();
    }

    using Energy   = Functor::Local::Energy_Functor<Cubic_Anisotropy>;
    using Gradient = Functor::Local::Gradient_Functor<Cubic_Anisotropy>;
    using Hessian  = Functor::Local::Hessian_Functor<Cubic_Anisotropy>;

    static std::size_t Sparse_Hessian_Size_per_Cell( const Data &, const Cache & )
    {
        return 0;
    };

    // Calculate the total energy for a single spin to be used in Monte Carlo.
    //      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
    using Energy_Single_Spin = Functor::Local::Energy_Single_Spin_Functor<Energy, 1>;

    // Interaction name as string
    static constexpr std::string_view name = "Cubic Anisotropy";

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
inline scalar Cubic_Anisotropy::Energy::operator()( const Index & index, const vectorfield & spins ) const
{
    using std::pow;
    scalar result = 0;
    if( !index.has_value() )
        return result;

    const auto & [ispin, iani] = *index;
    return -0.5 * data.magnitudes[iani]
           * ( pow( spins[ispin][0], 4.0 ) + pow( spins[ispin][1], 4.0 ) + pow( spins[ispin][2], 4.0 ) );
}

template<>
inline Vector3 Cubic_Anisotropy::Gradient::operator()( const Index & index, const vectorfield & spins ) const
{
    using std::pow;
    Vector3 result = Vector3::Zero();
    if( !index.has_value() )
        return result;

    const auto & [ispin, iani] = *index;

    for( int icomp = 0; icomp < 3; ++icomp )
    {
        result[icomp] = -2.0 * data.magnitudes[iani] * pow( spins[ispin][icomp], 3.0 );
    }
    return result;
}

template<>
template<typename F>
void Cubic_Anisotropy::Hessian::operator()( const Index & index, const vectorfield & spins, F & f ) const
{
    // TODO: Not yet implemented
}

} // namespace Interaction

} // namespace Spin

} // namespace Engine
#endif
