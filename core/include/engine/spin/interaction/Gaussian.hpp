#pragma once
#ifndef SPIRIT_CORE_ENGINE_INTERACTION_GAUSSIAN_HPP
#define SPIRIT_CORE_ENGINE_INTERACTION_GAUSSIAN_HPP

#include <engine/spin/interaction/ABC.hpp>

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
    using state_t = vectorfield;

    struct Data
    {
        // Parameters of the energy landscape
        scalarfield amplitude;
        scalarfield width;
        vectorfield center;
    };

    struct Cache
    {
        std::size_t n_gaussians;
    };

    static bool is_contributing( const Data & data, const Cache & )
    {
        return !data.amplitude.empty();
    };

    typedef int IndexType;

    using Index = std::optional<int>;

    static void clearIndex( Index & index )
    {
        index.reset();
    };

    using Energy   = Local::Energy_Functor<Gaussian>;
    using Gradient = Local::Gradient_Functor<Gaussian>;
    using Hessian  = Local::Hessian_Functor<Gaussian>;

    static std::size_t Sparse_Hessian_Size_per_Cell( const Data & data, const Cache & )
    {
        return 9 * data.amplitude.size();
    };

    // Calculate the total energy for a single spin
    using Energy_Single_Spin = Local::Energy_Single_Spin_Functor<Energy, 1>;

    // Interaction name as string
    static constexpr std::string_view name = "Gaussian";

    template<typename IndexVector>
    static void
    applyGeometry( const ::Data::Geometry & geometry, const intfield &, const Data &, Cache &, IndexVector & indices )
    {
        const auto N = geometry.nos;

#pragma omp parallel for
        for( int icell = 0; icell < geometry.n_cells_total; ++icell )
        {
            for( int ibasis = 0; ibasis < N; ++ibasis )
            {
                const int ispin                   = icell * N + ibasis;
                std::get<Index>( indices[ispin] ) = ispin;
            };
        }
    }
};

template<>
template<typename F>
void Gaussian::Hessian::operator()( const Index & index, const vectorfield & spins, F & f ) const
{
    if( !index.has_value() )
        return;

    const int ispin = *index;
    // Calculate Hessian
    for( unsigned int igauss = 0; igauss < data.amplitude.size(); ++igauss )
    {
        // Distance between spin and gaussian center
        scalar l = 1 - data.center[igauss].dot( spins[ispin] );
        // Prefactor for all alpha, beta
        scalar prefactor
            = data.amplitude[igauss] * std::exp( -std::pow( l, 2 ) / ( 2.0 * std::pow( data.width[igauss], 2 ) ) )
              / std::pow( data.width[igauss], 2 ) * ( std::pow( l, 2 ) / std::pow( data.width[igauss], 2 ) - 1 );
        // Effective Field contribution
        for( std::uint8_t alpha = 0; alpha < 3; ++alpha )
        {
            for( std::uint8_t beta = 0; beta < 3; ++beta )
            {
                std::size_t i = 3 * ispin + alpha;
                std::size_t j = 3 * ispin + beta;
                f( i, j, prefactor * data.center[igauss][alpha] * data.center[igauss][beta] );
            }
        }
    }
}

} // namespace Interaction

} // namespace Spin

} // namespace Engine

#endif
