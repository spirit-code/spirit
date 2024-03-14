#include <engine/spin/interaction/DMI.hpp>
#include <utility/Constants.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

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
scalar DMI::Energy::operator()( const Index & index, const vectorfield & spins ) const
{
    return std::transform_reduce(
        begin( index ), end( index ), scalar( 0.0 ), std::plus<scalar>{},
        [this, &spins]( const auto & idx ) -> scalar
        {
            const auto & [ispin, jspin, i_pair] = idx;
            return -0.5 * cache.magnitudes[i_pair]
                   * cache.normals[i_pair].dot( spins[ispin].cross( spins[jspin] ) );
        } );
}

template<>
Vector3 DMI::Gradient::operator()( const Index & index, const vectorfield & spins ) const
{
    return std::transform_reduce(
        begin( index ), end( index ), Vector3{ 0.0, 0.0, 0.0 }, std::plus<Vector3>{},
        [this, &spins]( const auto & idx ) -> Vector3
        {
            const auto & [ispin, jspin, i_pair] = idx;
            return -cache.magnitudes[i_pair] * spins[jspin].cross( cache.normals[i_pair] );
        } );
};

// Calculate the total energy for a single spin to be used in Monte Carlo.
//      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
// scalar DMI::Energy_Single_Spin( int ispin, const vectorfield & spins )
// {
//     const auto & geometry            = getGeometry();
//     const auto & boundary_conditions = getBoundaryConditions();
//
//     int icell  = ispin / geometry.n_cell_atoms;
//     int ibasis = ispin - icell * geometry.n_cell_atoms;
//
//     Pair pair_inv{};
//     scalar Energy = 0;
//     for( unsigned int ipair = 0; ipair < dmi_pairs.size(); ++ipair )
//     {
//         const auto & pair = dmi_pairs[ipair];
//         if( pair.i == ibasis )
//         {
//             int jspin = idx_from_pair(
//                 ispin, boundary_conditions, geometry.n_cells, geometry.n_cell_atoms, geometry.atom_types, pair );
//             if( jspin >= 0 )
//                 Energy
//                     -= this->dmi_magnitudes[ipair] * this->dmi_normals[ipair].dot( spins[ispin].cross( spins[jspin] ) );
//         }
// #if !( defined( SPIRIT_USE_OPENMP ) || defined( SPIRIT_USE_CUDA ) )
//         if( pair.j == ibasis )
//         {
//             const auto & t = pair.translations;
//             pair_inv       = Pair{ pair.j, pair.i, { -t[0], -t[1], -t[2] } };
//             int jspin      = idx_from_pair(
//                 ispin, boundary_conditions, geometry.n_cells, geometry.n_cell_atoms, geometry.atom_types, pair_inv );
//             if( jspin >= 0 )
//                 Energy
//                     += this->dmi_magnitudes[ipair] * this->dmi_normals[ipair].dot( spins[ispin].cross( spins[jspin] ) );
//         }
// #endif
//     }
//     return Energy;
// };

} // namespace Interaction

} // namespace Spin

} // namespace Engine
