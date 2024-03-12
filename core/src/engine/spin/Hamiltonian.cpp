#include <data/Spin_System.hpp>
#include <engine/Backend_par.hpp>
#include <engine/Indexing.hpp>
#include <engine/Neighbours.hpp>
#include <engine/Vectormath.hpp>
#include <engine/spin/Hamiltonian.hpp>
#include <utility/Constants.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <algorithm>

using namespace Data;
using namespace Utility;

using Engine::Indexing::check_atom_type;

namespace Engine
{

namespace Spin
{

// template<typename FirstInteraction, typename... Interactions>
// void Hamiltonian<FirstInteraction, Interactions...>::Energy_per_Spin( const vectorfield & spins, scalarfield & energy_per_spin )
// {
//     const auto nos = spins.size();
//     // Allocate if not already allocated
//     if( energy_per_spin.size() != nos )
//         energy_per_spin = scalarfield( nos, 0 );
//     // Otherwise set to zero
//     else
//         Vectormath::fill( energy_per_spin, 0 );
//
//     apply_active( [&spins, &energy_per_spin]( auto & interaction )
//                   { interaction->Energy_per_Spin( spins, energy_per_spin ); } );
// }
//
// template<typename FirstInteraction, typename... Interactions>
// void Hamiltonian<FirstInteraction, Interactions...>::Energy_Contributions_per_Spin(
//     const vectorfield & spins, vectorlabeled<scalarfield> & contributions )
// {
//     const auto nos                      = spins.size();
//     const auto active_interactions_size = active_count();
//
//     if( contributions.size() != active_interactions_size )
//     {
//         contributions = std::vector( active_interactions_size, std::pair{ std::string_view{}, scalarfield( nos, 0 ) } );
//     }
//     else
//     {
//         for( auto & contrib : contributions )
//         {
//             // Allocate if not already allocated
//             if( contrib.second.size() != nos )
//                 contrib.second = scalarfield( nos, 0 );
//             // Otherwise set to zero
//             else
//                 Vectormath::fill( contrib.second, 0 );
//         }
//     }
//
//     apply_active(
//         [&spins, &contributions, idx = 0]( auto & interaction ) mutable
//         {
//             contributions[idx].first = interaction->Name();
//             interaction->Energy_per_Spin( spins, contributions[idx].second );
//             ++idx;
//         } );
// }
//
// template<typename FirstInteraction, typename... Interactions>
// scalar Hamiltonian<FirstInteraction, Interactions...>::Energy_Single_Spin( int ispin, const vectorfield & spins )
// {
//     scalar energy = 0;
//     if( check_atom_type( this->geometry->atom_types[ispin] ) )
//     {
//         apply_active( [&spins, &energy]( auto & interaction ) { energy += interaction->Energy_Single_Spin( spins ); } );
//     }
//     return energy;
// }
//
// template<typename FirstInteraction, typename... Interactions>
// void Hamiltonian<FirstInteraction, Interactions...>::Gradient( const vectorfield & spins, vectorfield & gradient )
// {
//     const auto nos = spins.size();
//     // Allocate if not already allocated
//     if( gradient.size() != nos )
//         gradient = vectorfield( nos, Vector3::Zero() );
//     // Otherwise set to zero
//     else
//         Vectormath::fill( gradient, Vector3::Zero() );
//
//     apply_active( [&spins, &gradient]( auto & interaction ) { interaction->Gradient( spins, gradient ); } );
// }
//
// template<typename FirstInteraction, typename... Interactions>
// void Hamiltonian<FirstInteraction, Interactions...>::Gradient_and_Energy(
//     const vectorfield & spins, vectorfield & gradient, scalar & energy )
// {
//     const auto nos = spins.size();
//     // Allocate if not already allocated
//     if( gradient.size() != nos )
//         gradient = vectorfield( nos, Vector3::Zero() );
//     // Otherwise set to zero
//     else
//         Vectormath::fill( gradient, Vector3::Zero() );
//     energy = 0;
//
//     const auto N              = spins.size();
//     const auto * s            = spins.data();
//     const auto * g            = gradient.data();
//     static constexpr scalar c = 1.0 / static_cast<scalar>( common_spin_order );
//
//     apply_common( [&spins, &gradient]( auto & interaction ) { interaction->Gradient( spins, gradient ); } );
//
//     energy += Backend::par::reduce( N, [s, g] SPIRIT_LAMBDA( int idx ) { return c * g[idx].dot( s[idx] ); } );
//
//     apply_uncommon(
//         [&spins, &gradient, &energy]( auto & interaction )
//         {
//             interaction->Gradient( spins, gradient );
//             energy += interaction->Energy( spins );
//         } );
// }
//
// template<typename FirstInteraction, typename... Interactions>
// void Hamiltonian<FirstInteraction, Interactions...>::Hessian( const vectorfield & spins, MatrixX & hessian )
// {
//     // --- Set to zero
//     hessian.setZero();
//
//     apply_active( [&spins, &hessian]( auto & interaction ) { interaction->Hessian( spins, hessian ); } );
// }
//
// template<typename FirstInteraction, typename... Interactions>
// void Hamiltonian<FirstInteraction, Interactions...>::Sparse_Hessian( const vectorfield & spins, SpMatrixX & hessian )
// {
//     std::size_t sparse_size_per_cell = 0;
//     apply_active( [&sparse_size_per_cell]( auto & interaction )
//                   { sparse_size_per_cell += interaction->Sparse_Hessian_Size_per_Cell(); } );
//
//     std::vector<Spin::Interaction::triplet> tripletList;
//     tripletList.reserve( geometry->n_cells_total * sparse_size_per_cell );
//
//     apply_active( [&spins, &tripletList]( auto & interaction ) { interaction->Sparse_Hessian( spins, tripletList ); } );
//
//     hessian.setFromTriplets( tripletList.begin(), tripletList.end() );
// }
//
// template<typename... Interactions>
// void Hamiltonian<Interactions...>::Hessian_FD( const vectorfield & spins, MatrixX & hessian )
// {
//     // This is a regular finite difference implementation (probably not very efficient)
//     // using the differences between gradient values (not function)
//     // see https://v8doc.sas.com/sashtml/ormp/chap5/sect28.htm
//
//     std::size_t nos = spins.size();
//
//     vectorfield spins_pi( nos );
//     vectorfield spins_mi( nos );
//     vectorfield spins_pj( nos );
//     vectorfield spins_mj( nos );
//
//     spins_pi = spins;
//     spins_mi = spins;
//     spins_pj = spins;
//     spins_mj = spins;
//
//     vectorfield grad_pi( nos );
//     vectorfield grad_mi( nos );
//     vectorfield grad_pj( nos );
//     vectorfield grad_mj( nos );
//
//     for( std::size_t i = 0; i < nos; ++i )
//     {
//         for( std::size_t j = 0; j < nos; ++j )
//         {
//             for( std::uint8_t alpha = 0; alpha < 3; ++alpha )
//             {
//                 for( std::uint8_t beta = 0; beta < 3; ++beta )
//                 {
//                     // Displace
//                     spins_pi[i][alpha] += delta;
//                     spins_mi[i][alpha] -= delta;
//                     spins_pj[j][beta] += delta;
//                     spins_mj[j][beta] -= delta;
//
//                     // Calculate Hessian component
//                     this->Gradient( spins_pi, grad_pi );
//                     this->Gradient( spins_mi, grad_mi );
//                     this->Gradient( spins_pj, grad_pj );
//                     this->Gradient( spins_mj, grad_mj );
//
//                     hessian( 3 * i + alpha, 3 * j + beta )
//                         = 0.25 / delta
//                           * ( grad_pj[i][alpha] - grad_mj[i][alpha] + grad_pi[j][beta] - grad_mi[j][beta] );
//
//                     // Un-Displace
//                     spins_pi[i][alpha] -= delta;
//                     spins_mi[i][alpha] += delta;
//                     spins_pj[j][beta] -= delta;
//                     spins_mj[j][beta] += delta;
//                 }
//             }
//         }
//     }
// }
//
// template<typename FirstInteraction, typename... Interactions>
// void Hamiltonian<FirstInteraction, Interactions...>::Gradient_FD( const vectorfield & spins, vectorfield & gradient )
// {
//     std::size_t nos = spins.size();
//
//     // Calculate finite difference
//     vectorfield spins_plus( nos );
//     vectorfield spins_minus( nos );
//
//     spins_plus  = spins;
//     spins_minus = spins;
//
//     for( std::size_t i = 0; i < nos; ++i )
//     {
//         for( std::uint8_t dim = 0; dim < 3; ++dim )
//         {
//             // Displace
//             spins_plus[i][dim] += delta;
//             spins_minus[i][dim] -= delta;
//
//             // Calculate gradient component
//             scalar E_plus    = this->Energy( spins_plus );
//             scalar E_minus   = this->Energy( spins_minus );
//             gradient[i][dim] = 0.5 * ( E_plus - E_minus ) / delta;
//
//             // Un-Displace
//             spins_plus[i][dim] -= delta;
//             spins_minus[i][dim] += delta;
//         }
//     }
// }
//
// template<typename FirstInteraction, typename... Interactions>
// scalar Hamiltonian<FirstInteraction, Interactions...>::Energy( const vectorfield & spins )
// {
//     scalar sum = 0;
//     apply_active( [&spins, &sum]( auto & interaction ) { sum += interaction->Energy( spins ); } );
//     return sum;
// }
//
// template<typename FirstInteraction, typename... Interactions>
// Data::vectorlabeled<scalar> Hamiltonian<FirstInteraction, Interactions...>::Energy_Contributions( const vectorfield & spins )
// {
//     vectorlabeled<scalar> contributions( 0 );
//     contributions.reserve( active_count() );
//     apply_active( [&contributions, &spins]( auto & interaction )
//                   { contributions.emplace_back( interaction->Name(), interaction->Energy( spins ) ); } );
//     return contributions;
// }

} // namespace Spin

} // namespace Engine
