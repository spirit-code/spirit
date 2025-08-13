#pragma once
#ifndef SPIRIT_CORE_ENGINE_INTERACTION_GAUSSIAN_HPP
#define SPIRIT_CORE_ENGINE_INTERACTION_GAUSSIAN_HPP

#include <engine/Index_Container.hpp>
#include <engine/spin/StateType.hpp>
#include <engine/spin/interaction/Functor_Prototypes.hpp>

#include <Eigen/Dense>

namespace Engine
{

namespace Spin
{

namespace Interaction
{

/*
The Gaussian Hamiltonian is meant for testing purposes and demonstrations. Spins do not interact.
A set of gaussians is summed with weight-factors so as to create an arbitrary energy landscape.
E = sum_i^N a_i exp( -l_i^2(m)/(2sigma_i^2) ) where l_i(m) is the distance of m to the gaussian i,
    a_i is the gaussian amplitude and sigma_i the width
*/
struct Gaussian
{
    using state_t = StateType;

    struct Data
    {
        // Parameters of the energy landscape
        scalarfield amplitude{};
        scalarfield width{};
        vectorfield center{};

        Data() = default;
        Data( scalarfield amplitude, scalarfield width, vectorfield center )
                : amplitude( std::move( amplitude ) ), width( std::move( width ) ), center( std::move( center ) ) {};
    };

    static bool valid_data( const Data & data )
    {
        return ( data.amplitude.size() == data.width.size() && data.amplitude.size() == data.center.size() );
    };

    struct Cache
    {
        std::size_t n_gaussians;
    };

    static bool is_contributing( const Data & data, const Cache & )
    {
        return !data.amplitude.empty();
    };

    struct Index
    {
        int ispin;
    };

    using IndexContainer = Engine::IndexContainer<Gaussian>;

    using Energy   = Functor::Local::Energy_Functor<Functor::Local::DataRef<Gaussian>>;
    using Gradient = Functor::Local::Gradient_Functor<Functor::Local::DataRef<Gaussian>>;
    using Hessian  = Functor::Local::Hessian_Functor<Functor::Local::DataRef<Gaussian>>;

    static std::size_t Sparse_Hessian_Size_per_Cell( const Data & data, const Cache & )
    {
        return 9 * data.amplitude.size();
    };

    // Calculate the total energy for a single spin
    using Energy_Single_Spin = Functor::Local::Energy_Single_Spin_Functor<Energy, 1>;

    // Interaction name as string
    static constexpr std::string_view name = "Gaussian";

    static void applyGeometry( const ::Data::Geometry & geometry, const Data &, Cache &, IndexContainer & container )
    {
        auto indices = std::vector( geometry.nos, field<Index>{} );

        for( int ispin = 0; ispin < geometry.nos; ++ispin )
            indices[ispin].push_back( Index{ ispin } );

        container = make_index_container<Gaussian>( std::move( indices ) );
    }
};

template<>
struct Functor::Local::DataRef<Gaussian>
{
    using Interaction = Gaussian;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;

    DataRef( const Data & data, const Cache & cache ) noexcept
            : is_contributing( Interaction::is_contributing( data, cache ) ),
              n_gaussians( data.amplitude.size() ),
              amplitude( data.amplitude.data() ),
              width( data.width.data() ),
              center( data.center.data() )
    {
    }

    const bool is_contributing;

protected:
    std::size_t n_gaussians;
    const scalar * amplitude;
    const scalar * width;
    const Vector3 * center;
};

template<>
inline scalar Gaussian::Energy::operator()( Span<const Index> index, quantity<const Vector3 *> state ) const
{

    if( !is_contributing )
        return 0;
    else
        return Backend::transform_reduce(
            index.begin(), index.end(), scalar( 0.0 ), Backend::plus<scalar>{},
            [this, state] SPIRIT_LAMBDA( const Index & idx ) -> scalar
            {
                scalar result = 0;
                for( unsigned int igauss = 0; igauss < n_gaussians; ++igauss )
                {
                    // Distance between spin and gaussian center
                    scalar l = 1 - center[igauss].dot( state.spin[idx.ispin] );
                    result += amplitude[igauss] * std::exp( -l * l / ( 2.0 * width[igauss] * width[igauss] ) );
                };
                return result;
            } );
}

template<>
inline Vector3 Gaussian::Gradient::operator()( Span<const Index> index, quantity<const Vector3 *> state ) const
{
    if( !is_contributing )
        return Vector3::Zero();
    else
        return Backend::transform_reduce(
            index.begin(), index.end(), Vector3{ 0.0, 0.0, 0.0 }, Backend::plus<Vector3>{},
            [this, state] SPIRIT_LAMBDA( const Index & idx ) -> Vector3
            {
                Vector3 result = Vector3::Zero();
                // Calculate gradient
                for( unsigned int i = 0; i < n_gaussians; ++i )
                {
                    // Scalar product of spin and gaussian center
                    scalar l = 1 - center[i].dot( state.spin[idx.ispin] );
                    // Prefactor
                    scalar prefactor = amplitude[i] * std::exp( -l * l / ( 2.0 * width[i] * width[i] ) ) * l
                                       / ( width[i] * width[i] );
                    // Gradient contribution
                    result += prefactor * center[i];
                }
                return result;
            } );
}

template<>
template<typename Callable>
void Gaussian::Hessian::operator()( Span<const Index> index, const StateType & state, Callable & hessian ) const
{
    if( !is_contributing )
        return;

    Backend::cpu::for_each(
        index.begin(), index.end(),
        [this, state, &hessian]( const Index & idx )
        {
            // Calculate Hessian
            for( unsigned int igauss = 0; igauss < n_gaussians; ++igauss )
            {
                // Distance between spin and gaussian center
                scalar l = 1 - center[igauss].dot( state.spin[idx.ispin] );
                // Prefactor for all alpha, beta
                scalar prefactor
                    = amplitude[igauss] * std::exp( -std::pow( l, 2 ) / ( 2.0 * std::pow( width[igauss], 2 ) ) )
                      / std::pow( width[igauss], 2 ) * ( std::pow( l, 2 ) / std::pow( width[igauss], 2 ) - 1 );
                // Effective Field contribution
                for( std::uint8_t alpha = 0; alpha < 3; ++alpha )
                {
                    for( std::uint8_t beta = 0; beta < 3; ++beta )
                    {
                        std::size_t i = 3 * idx.ispin + alpha;
                        std::size_t j = 3 * idx.ispin + beta;
                        hessian( i, j, prefactor * center[igauss][alpha] * center[igauss][beta] );
                    }
                }
            }
        } );
}

} // namespace Interaction

} // namespace Spin

} // namespace Engine

#endif
