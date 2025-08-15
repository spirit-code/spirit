#pragma once
#ifndef SPIRIT_CORE_ENGINE_INTERACTION_CUBIC_ANISOTROPY_HPP
#define SPIRIT_CORE_ENGINE_INTERACTION_CUBIC_ANISOTROPY_HPP

#include <engine/Index_Container.hpp>
#include <engine/Indexing.hpp>
#include <engine/spin/StateType.hpp>
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
    using state_t = StateType;

    struct Data
    {
        intfield indices{};
        scalarfield magnitudes{};

        Data() = default;
        Data( intfield indices, scalarfield magnitudes )
                : indices( std::move( indices ) ), magnitudes( std::move( magnitudes ) ) {};
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

    struct Index
    {
        int ispin, iani;
    };

    using IndexContainer = Engine::IndexContainer<Cubic_Anisotropy>;

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

    static void
    applyGeometry( const ::Data::Geometry & geometry, const Data & data, Cache &, IndexContainer & container )
    {
        using Indexing::check_atom_type;
        auto indices = std::vector( geometry.nos, field<Index>{} );

        for( int icell = 0; icell < geometry.n_cells_total; ++icell )
        {
            for( int iani = 0; iani < data.indices.size(); ++iani )
            {
                int ispin = icell * geometry.n_cell_atoms + data.indices[iani];
                if( check_atom_type( geometry.atom_types[ispin] ) )
                    indices[ispin].push_back( Index{ ispin, iani } );
            }
        }

        container = make_index_container<Cubic_Anisotropy>( std::move( indices ) );
    };
};

template<>
struct Functor::Local::DataRef<Cubic_Anisotropy>
{
    using Interaction = Cubic_Anisotropy;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;

    DataRef( const Data & data, const Cache & cache ) noexcept
            : is_contributing( Interaction::is_contributing( data, cache ) ), magnitudes( data.magnitudes.data() )
    {
    }

    const bool is_contributing;

protected:
    const scalar * magnitudes;
};

template<>
inline scalar Cubic_Anisotropy::Energy::operator()( Span<const Index> index, quantity<const Vector3 *> state ) const
{
    using Utility::fastpow;
    if( !is_contributing )
        return 0;
    else
        return Backend::transform_reduce(
            index.begin(), index.end(), scalar( 0.0 ), Backend::plus<scalar>{},
            [this, state] SPIRIT_LAMBDA( const Index & idx ) -> scalar
            {
                return -0.5 * magnitudes[idx.iani]
                       * ( fastpow( state.spin[idx.ispin][0], 4u ) + fastpow( state.spin[idx.ispin][1], 4u )
                           + fastpow( state.spin[idx.ispin][2], 4u ) );
            } );
}

template<>
inline Vector3 Cubic_Anisotropy::Gradient::operator()( Span<const Index> index, quantity<const Vector3 *> state ) const
{
    using Utility::fastpow;
    if( !is_contributing )
        return Vector3::Zero();
    else
        return Backend::transform_reduce(
            index.begin(), index.end(), Vector3{ Vector3::Zero() }, Backend::plus<Vector3>{},
            [this, state] SPIRIT_LAMBDA( const Index & idx ) -> Vector3
            {
                Vector3 result = Vector3::Zero();
                for( int icomp = 0; icomp < 3; ++icomp )
                {
                    result[icomp] = -2.0 * magnitudes[idx.iani] * fastpow( state.spin[idx.ispin][icomp], 3u );
                }
                return result;
            } );
}

template<>
template<typename Callable>
void Cubic_Anisotropy::Hessian::operator()( Span<const Index> index, const StateType & state, Callable & hessian ) const
{
    using Utility::fastpow;
    Backend::cpu::for_each(
        index.begin(), index.end(),
        [this, &state, &index, &hessian]( const Index & idx )
        {
            for( int alpha = 0; alpha < 3; ++alpha )
            {
                const int i = 3 * idx.ispin + alpha;
                const int j = 3 * idx.ispin + alpha;

                hessian( i, j, -6.0 * magnitudes[idx.iani] * fastpow( state.spin[idx.ispin][alpha], 2u ) );
            }
        } );
}

} // namespace Interaction

} // namespace Spin

} // namespace Engine
#endif
