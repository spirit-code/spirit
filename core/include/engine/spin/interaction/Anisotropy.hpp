#pragma once
#ifndef SPIRIT_CORE_ENGINE_INTERACTION_ANISOTROPY_HPP
#define SPIRIT_CORE_ENGINE_INTERACTION_ANISOTROPY_HPP

#include <engine/Index_Container.hpp>
#include <engine/Indexing.hpp>
#include <engine/spin/StateType.hpp>
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
    using state_t = StateType;

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

    struct Index
    {
        int ispin, iani;
    };

    using IndexContainer = Engine::IndexContainer<Anisotropy>;

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

        container = make_index_container<Anisotropy>( std::move( indices ) );
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
inline scalar Anisotropy::Energy::operator()( Span<const Index> index, quantity<const Vector3 *> state ) const
{
    return -1.0
           * Backend::transform_reduce(
               index.begin(), index.end(), scalar( 0 ), Backend::plus<scalar>{},
               [this, state] SPIRIT_LAMBDA( const Index & idx ) -> scalar
               {
                   const auto d = normals[idx.iani].dot( state.spin[idx.ispin] );
                   return magnitudes[idx.iani] * d * d;
               } );
}

template<>
inline Vector3 Anisotropy::Gradient::operator()( Span<const Index> index, quantity<const Vector3 *> state ) const
{
    return -2.0
           * Backend::transform_reduce(
               index.begin(), index.end(), Vector3( Vector3::Zero() ), Backend::plus<Vector3>{},
               [this, state] SPIRIT_LAMBDA( const Index & idx ) -> Vector3
               { return magnitudes[idx.iani] * normals[idx.iani].dot( state.spin[idx.ispin] ) * normals[idx.iani]; } );
}

template<>
template<typename Callable>
void Anisotropy::Hessian::operator()( Span<const Index> index, const StateType &, Callable & hessian ) const
{
    Backend::cpu::for_each(
        index.begin(), index.end(),
        [this, &index, &hessian]( const Index & idx )
        {
            for( int alpha = 0; alpha < 3; ++alpha )
            {
                for( int beta = 0; beta < 3; ++beta )
                {
                    const int i = 3 * idx.ispin + alpha;
                    const int j = 3 * idx.ispin + alpha;

                    hessian( i, j, -2.0 * magnitudes[idx.iani] * normals[idx.iani][alpha] * normals[idx.iani][beta] );
                }
            }
        } );
}

} // namespace Interaction

} // namespace Spin

} // namespace Engine
#endif
