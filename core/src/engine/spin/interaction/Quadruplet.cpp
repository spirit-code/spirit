#include <engine/Backend_par.hpp>
#include <engine/Indexing.hpp>
#include <engine/Neighbours.hpp>
#include <engine/Vectormath.hpp>
#include <engine/spin/interaction/Quadruplet.hpp>
#include <utility/Constants.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <utility>

#ifndef SPIRIT_USE_CUDA
#include <algorithm>
#else
#include <complex> // TODO: check if I need complex for the CUDA implementation
#endif

using namespace Data;
using namespace Utility;
namespace C = Utility::Constants;
using Engine::Indexing::check_atom_type;
using Engine::Indexing::idx_from_pair;
using Engine::Indexing::idx_from_translations;

#ifdef SPIRIT_USE_CUDA
using Engine::Indexing::cu_check_atom_type;
using Engine::Indexing::cu_idx_from_pair;
using Engine::Indexing::cu_tupel_from_idx;
#endif

namespace Engine
{

namespace Spin
{

namespace Interaction
{

template<>
void Quadruplet::Energy::operator()( const vectorfield & spins, scalarfield & energy ) const
{
    Quadruplet::apply(
        [&spins, &energy, &magnitudes = data.magnitudes](
            const auto iquad, const auto ispin, const auto jspin, const auto kspin, const auto lspin )
        {
            if( ispin >= 0 && jspin >= 0 && kspin >= 0 && lspin >= 0 )
            {
                const scalar quad_energy = 0.25 * magnitudes[iquad] * ( spins[ispin].dot( spins[jspin] ) )
                                           * ( spins[kspin].dot( spins[lspin] ) );
                energy[ispin] -= quad_energy;
                energy[jspin] -= quad_energy;
                energy[kspin] -= quad_energy;
                energy[lspin] -= quad_energy;
            }
        },
        data, cache );
}

// Calculate the total energy for a single spin to be used in Monte Carlo.
//      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
template<>
scalar Quadruplet::Energy_Single_Spin::operator()( const int ispin, const vectorfield & spins ) const
{
    // TODO
    return 0;
};

template<>
void Quadruplet::Gradient::operator()( const vectorfield & spins, vectorfield & gradient ) const
{
    Quadruplet::apply(
        [&spins, &gradient, &magnitudes = data.magnitudes](
            const auto iquad, const auto ispin, const auto jspin, const auto kspin, const auto lspin )
        {
            if( ispin >= 0 && jspin >= 0 && kspin >= 0 && lspin >= 0 )
            {
                gradient[ispin] -= magnitudes[iquad] * spins[jspin] * ( spins[kspin].dot( spins[lspin] ) );
                gradient[jspin] -= magnitudes[iquad] * spins[ispin] * ( spins[kspin].dot( spins[lspin] ) );
                gradient[kspin] -= magnitudes[iquad] * ( spins[ispin].dot( spins[jspin] ) ) * spins[lspin];
                gradient[lspin] -= magnitudes[iquad] * ( spins[ispin].dot( spins[jspin] ) ) * spins[kspin];
            }
        },
        data, cache );
};

} // namespace Interaction

} // namespace Spin

} // namespace Engine
