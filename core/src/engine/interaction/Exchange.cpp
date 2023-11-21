#include <engine/Backend_par.hpp>
#include <engine/Indexing.hpp>
#include <engine/Neighbours.hpp>
#include <engine/Vectormath.hpp>
#include <engine/interaction/Exchange.hpp>
#include <utility/Constants.hpp>
#include <utility>

#include <Eigen/Core>
#include <Eigen/Dense>

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

namespace Interaction
{

Exchange::Exchange( Hamiltonian * hamiltonian, pairfield pairs, scalarfield magnitudes ) noexcept
        : Interaction::Base<Exchange>( hamiltonian, scalarfield( 0 ) ),
          exchange_pairs_in( std::move( pairs ) ),
          exchange_magnitudes_in( std::move( magnitudes ) )
{
    this->updateGeometry();
}

Exchange::Exchange( Hamiltonian * hamiltonian, const ScalarPairfieldData & exchange ) noexcept
        : Exchange( hamiltonian, exchange.pairs, exchange.magnitudes )
{
}

Exchange::Exchange( Hamiltonian * hamiltonian, const scalarfield & shell_magnitudes ) noexcept
        : Interaction::Base<Exchange>( hamiltonian, scalarfield( 0 ) ), exchange_shell_magnitudes( shell_magnitudes )
{
    this->updateGeometry();
}

bool Exchange::is_contributing() const
{
    return !exchange_pairs.empty();
}

void Exchange::updateFromGeometry( const Geometry * geometry )
{
    this->exchange_pairs      = pairfield( 0 );
    this->exchange_magnitudes = scalarfield( 0 );
    if( !exchange_shell_magnitudes.empty() )
    {
        // Generate Exchange neighbours
        intfield exchange_shells( 0 );
        Neighbours::Get_Neighbours_in_Shells(
            *geometry, exchange_shell_magnitudes.size(), exchange_pairs, exchange_shells, use_redundant_neighbours );
        for( std::size_t ipair = 0; ipair < exchange_pairs.size(); ++ipair )
        {
            this->exchange_magnitudes.push_back( exchange_shell_magnitudes[exchange_shells[ipair]] );
        }
    }
    else
    {
        // Use direct list of pairs
        this->exchange_pairs      = this->exchange_pairs_in;
        this->exchange_magnitudes = this->exchange_magnitudes_in;
        if( use_redundant_neighbours )
        {
            for( std::size_t i = 0; i < exchange_pairs_in.size(); ++i )
            {
                auto & p = exchange_pairs_in[i];
                auto & t = p.translations;
                this->exchange_pairs.emplace_back( Pair{ p.j, p.i, { -t[0], -t[1], -t[2] } } );
                this->exchange_magnitudes.push_back( exchange_magnitudes_in[i] );
            }
        }
    }
}

#ifdef SPIRIT_USE_CUDA
__global__ void CU_E_Exchange(
    const Vector3 * spins, const int * atom_types, const int * boundary_conditions, const int * n_cells,
    int n_cell_atoms, int n_pairs, const Pair * pairs, const scalar * magnitudes, scalar * energy, size_t size )
{
    int bc[3] = { boundary_conditions[0], boundary_conditions[1], boundary_conditions[2] };
    int nc[3] = { n_cells[0], n_cells[1], n_cells[2] };

    for( auto icell = blockIdx.x * blockDim.x + threadIdx.x; icell < size; icell += blockDim.x * gridDim.x )
    {
        for( auto ipair = 0; ipair < n_pairs; ++ipair )
        {
            int ispin = pairs[ipair].i + icell * n_cell_atoms;
            int jspin = cu_idx_from_pair( ispin, bc, nc, n_cell_atoms, atom_types, pairs[ipair] );
            if( jspin >= 0 )
            {
                energy[ispin] -= 0.5 * magnitudes[ipair] * spins[ispin].dot( spins[jspin] );
            }
        }
    }
}
#endif

void Exchange::Energy_per_Spin( const vectorfield & spins, scalarfield & energy )
{
    const auto * geometry            = hamiltonian->geometry.get();
    const auto & boundary_conditions = hamiltonian->boundary_conditions;

#ifdef SPIRIT_USE_CUDA
    int size = geometry->n_cells_total;
    CU_E_Exchange<<<( size + 1023 ) / 1024, 1024>>>(
        spins.data(), geometry->atom_types.data(), boundary_conditions.data(), geometry->n_cells.data(),
        geometry->n_cell_atoms, this->exchange_pairs.size(), this->exchange_pairs.data(),
        this->exchange_magnitudes.data(), energy.data(), size );
    CU_CHECK_AND_SYNC();
#else

#pragma omp parallel for
    for( unsigned int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        for( unsigned int i_pair = 0; i_pair < exchange_pairs.size(); ++i_pair )
        {
            int ispin = exchange_pairs[i_pair].i + icell * geometry->n_cell_atoms;
            int jspin = idx_from_pair(
                ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                exchange_pairs[i_pair] );
            if( jspin >= 0 )
            {
                energy[ispin] -= 0.5 * exchange_magnitudes[i_pair] * spins[ispin].dot( spins[jspin] );
#ifndef SPIRIT_USE_OPENMP
                energy[jspin] -= 0.5 * exchange_magnitudes[i_pair] * spins[ispin].dot( spins[jspin] );
#endif
            }
        }
    }
#endif
}

// Calculate the total energy for a single spin to be used in Monte Carlo.
//      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
scalar Exchange::Energy_Single_Spin( int ispin, const vectorfield & spins )
{
    const auto * geometry            = hamiltonian->geometry.get();
    const auto & boundary_conditions = hamiltonian->boundary_conditions;

    int icell  = ispin / geometry->n_cell_atoms;
    int ibasis = ispin - icell * geometry->n_cell_atoms;

    Pair pair_inv{};
    scalar Energy = 0;
    for( unsigned int ipair = 0; ipair < exchange_pairs.size(); ++ipair )
    {
        const auto & pair = exchange_pairs[ipair];
        if( pair.i == ibasis )
        {
            int jspin = idx_from_pair(
                ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, pair );
            if( jspin >= 0 )
                Energy -= this->exchange_magnitudes[ipair] * spins[ispin].dot( spins[jspin] );
        }
#if !( defined( SPIRIT_USE_OPENMP ) || defined( SPIRIT_USE_CUDA ) )
        if( pair.j == ibasis )
        {
            const auto & t = pair.translations;
            pair_inv       = Pair{ pair.j, pair.i, { -t[0], -t[1], -t[2] } };
            int jspin      = idx_from_pair(
                ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, pair_inv );
            if( jspin >= 0 )
                Energy -= this->exchange_magnitudes[ipair] * spins[ispin].dot( spins[jspin] );
        }
#endif
    }
    return Energy;
};

void Exchange::Hessian( const vectorfield & spins, MatrixX & hessian )
{
    const auto * geometry            = hamiltonian->geometry.get();
    const auto & boundary_conditions = hamiltonian->boundary_conditions;

#pragma omp parallel for
    for( int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        for( unsigned int i_pair = 0; i_pair < exchange_pairs.size(); ++i_pair )
        {
            int ispin = exchange_pairs[i_pair].i + icell * geometry->n_cell_atoms;
            int jspin = idx_from_pair(
                ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                exchange_pairs[i_pair] );
            if( jspin >= 0 )
            {
                for( int alpha = 0; alpha < 3; ++alpha )
                {
                    int i = 3 * ispin + alpha;
                    int j = 3 * jspin + alpha;

                    hessian( i, j ) += -exchange_magnitudes[i_pair];
#if !( defined( SPIRIT_USE_OPENMP ) || defined( SPIRIT_USE_CUDA ) )
                    hessian( j, i ) += -exchange_magnitudes[i_pair];
#endif
                }
            }
        }
    }
};

void Exchange::Sparse_Hessian( const vectorfield & spins, std::vector<triplet> & hessian )
{
    const auto * geometry            = hamiltonian->geometry.get();
    const auto & boundary_conditions = hamiltonian->boundary_conditions;

    for( int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        for( unsigned int i_pair = 0; i_pair < exchange_pairs.size(); ++i_pair )
        {
            int ispin = exchange_pairs[i_pair].i + icell * geometry->n_cell_atoms;
            int jspin = idx_from_pair(
                ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                exchange_pairs[i_pair] );
            if( jspin >= 0 )
            {
                for( int alpha = 0; alpha < 3; ++alpha )
                {
                    int i = 3 * ispin + alpha;
                    int j = 3 * jspin + alpha;

                    hessian.emplace_back( i, j, -exchange_magnitudes[i_pair] );
#if !( defined( SPIRIT_USE_OPENMP ) || defined( SPIRIT_USE_CUDA ) )
                    hessian.emplace_back( j, i, -exchange_magnitudes[i_pair] );
#endif
                }
            }
        }
    }
};

#ifdef SPIRIT_USE_CUDA
__global__ void CU_Gradient_Exchange(
    const Vector3 * spins, const int * atom_types, const int * boundary_conditions, const int * n_cells,
    int n_cell_atoms, int n_pairs, const Pair * pairs, const scalar * magnitudes, Vector3 * gradient, size_t size )
{
    int bc[3] = { boundary_conditions[0], boundary_conditions[1], boundary_conditions[2] };
    int nc[3] = { n_cells[0], n_cells[1], n_cells[2] };

    for( auto icell = blockIdx.x * blockDim.x + threadIdx.x; icell < size; icell += blockDim.x * gridDim.x )
    {
        for( auto ipair = 0; ipair < n_pairs; ++ipair )
        {
            int ispin = pairs[ipair].i + icell * n_cell_atoms;
            int jspin = cu_idx_from_pair( ispin, bc, nc, n_cell_atoms, atom_types, pairs[ipair] );
            if( jspin >= 0 )
            {
                gradient[ispin] -= magnitudes[ipair] * spins[jspin];
            }
        }
    }
}
#endif

void Exchange::Gradient( const vectorfield & spins, vectorfield & gradient )
{
    const auto * geometry            = hamiltonian->geometry.get();
    const auto & boundary_conditions = hamiltonian->boundary_conditions;

#ifdef SPIRIT_USE_CUDA
    int size = geometry->n_cells_total;
    CU_Gradient_Exchange<<<( size + 1023 ) / 1024, 1024>>>(
        spins.data(), geometry->atom_types.data(), boundary_conditions.data(), geometry->n_cells.data(),
        geometry->n_cell_atoms, this->exchange_pairs.size(), this->exchange_pairs.data(),
        this->exchange_magnitudes.data(), gradient.data(), size );
    CU_CHECK_AND_SYNC();
#else

#pragma omp parallel for
    for( int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        for( unsigned int i_pair = 0; i_pair < exchange_pairs.size(); ++i_pair )
        {
            int ispin = exchange_pairs[i_pair].i + icell * geometry->n_cell_atoms;
            int jspin = idx_from_pair(
                ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                exchange_pairs[i_pair] );
            if( jspin >= 0 )
            {
                gradient[ispin] -= exchange_magnitudes[i_pair] * spins[jspin];
#ifndef SPIRIT_USE_OPENMP
                gradient[jspin] -= exchange_magnitudes[i_pair] * spins[ispin];
#endif
            }
        }
    }
#endif
};

} // namespace Interaction

} // namespace Engine
