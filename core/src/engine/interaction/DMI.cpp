#include <engine/Backend_par.hpp>
#include <engine/Indexing.hpp>
#include <engine/Neighbours.hpp>
#include <engine/Vectormath.hpp>
#include <engine/interaction/DMI.hpp>
#include <utility/Constants.hpp>

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

DMI::DMI( Hamiltonian * hamiltonian, pairfield pairs, scalarfield magnitudes, vectorfield normals ) noexcept
        : Interaction::Base<DMI>( hamiltonian, scalarfield( 0 ) ),
          dmi_shell_magnitudes( 0 ),
          dmi_shell_chirality( 0 ),
          dmi_pairs_in( std::move( pairs ) ),
          dmi_magnitudes_in( std::move( magnitudes ) ),
          dmi_normals_in( std::move( normals ) )
{
    this->updateGeometry();
}

DMI::DMI( Hamiltonian * hamiltonian, const Data::VectorPairfieldData & dmi ) noexcept
        : DMI( hamiltonian, dmi.pairs, dmi.magnitudes, dmi.normals )
{
}

DMI::DMI( Hamiltonian * hamiltonian, scalarfield shell_magnitudes, const int chirality ) noexcept
        : Interaction::Base<DMI>( hamiltonian, scalarfield( 0 ) ),
          dmi_shell_magnitudes( std::move( shell_magnitudes ) ),
          dmi_shell_chirality( chirality ),
          dmi_pairs_in( 0 ),
          dmi_magnitudes_in( 0 ),
          dmi_normals_in( 0 )
{
    this->updateGeometry();
}

bool DMI::is_contributing() const
{
    return !dmi_pairs.empty();
}

void DMI::updateFromGeometry( const Geometry * geometry )
{
    this->dmi_pairs      = pairfield( 0 );
    this->dmi_magnitudes = scalarfield( 0 );
    this->dmi_normals    = vectorfield( 0 );
    if( !dmi_shell_magnitudes.empty() )
    {
        // Generate DMI neighbours and normals
        intfield dmi_shells( 0 );
        Neighbours::Get_Neighbours_in_Shells(
            *geometry, dmi_shell_magnitudes.size(), dmi_pairs, dmi_shells, use_redundant_neighbours );
        for( std::size_t ineigh = 0; ineigh < dmi_pairs.size(); ++ineigh )
        {
            this->dmi_normals.push_back(
                Neighbours::DMI_Normal_from_Pair( *geometry, dmi_pairs[ineigh], this->dmi_shell_chirality ) );
            this->dmi_magnitudes.push_back( dmi_shell_magnitudes[dmi_shells[ineigh]] );
        }
    }
    else
    {
        // Use direct list of pairs
        this->dmi_pairs      = this->dmi_pairs_in;
        this->dmi_magnitudes = this->dmi_magnitudes_in;
        this->dmi_normals    = this->dmi_normals_in;
        if( use_redundant_neighbours )
        {
            for( std::size_t i = 0; i < dmi_pairs_in.size(); ++i )
            {
                auto & p = dmi_pairs_in[i];
                auto & t = p.translations;
                this->dmi_pairs.emplace_back( Pair{ p.j, p.i, { -t[0], -t[1], -t[2] } } );
                this->dmi_magnitudes.emplace_back( dmi_magnitudes_in[i] );
                this->dmi_normals.emplace_back( -dmi_normals_in[i] );
            }
        }
    }
}

#ifdef SPIRIT_USE_CUDA
__global__ void CU_E_DMI(
    const Vector3 * spins, const int * atom_types, const int * boundary_conditions, const int * n_cells,
    int n_cell_atoms, int n_pairs, const Pair * pairs, const scalar * magnitudes, const Vector3 * normals,
    scalar * energy, size_t size )
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
                energy[ispin] -= 0.5 * magnitudes[ipair] * normals[ipair].dot( spins[ispin].cross( spins[jspin] ) );
            }
        }
    }
}
#endif

void DMI::Energy_per_Spin( const vectorfield & spins, scalarfield & energy )
{
    const auto * geometry            = hamiltonian->geometry.get();
    const auto & boundary_conditions = hamiltonian->boundary_conditions;

#ifdef SPIRIT_USE_CUDA
    int size = geometry->n_cells_total;
    CU_E_DMI<<<( size + 1023 ) / 1024, 1024>>>(
        spins.data(), geometry->atom_types.data(), boundary_conditions.data(), geometry->n_cells.data(),
        geometry->n_cell_atoms, this->dmi_pairs.size(), this->dmi_pairs.data(), this->dmi_magnitudes.data(),
        this->dmi_normals.data(), energy.data(), size );
    CU_CHECK_AND_SYNC();
#else
#pragma omp parallel for
    for( unsigned int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        for( unsigned int i_pair = 0; i_pair < dmi_pairs.size(); ++i_pair )
        {
            int ispin = dmi_pairs[i_pair].i + icell * geometry->n_cell_atoms;
            int jspin = idx_from_pair(
                ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                dmi_pairs[i_pair] );
            if( jspin >= 0 )
            {
                energy[ispin]
                    -= 0.5 * dmi_magnitudes[i_pair] * dmi_normals[i_pair].dot( spins[ispin].cross( spins[jspin] ) );
#ifndef SPIRIT_USE_OPENMP
                energy[jspin]
                    -= 0.5 * dmi_magnitudes[i_pair] * dmi_normals[i_pair].dot( spins[ispin].cross( spins[jspin] ) );
#endif
            }
        }
    }
#endif
}

// Calculate the total energy for a single spin to be used in Monte Carlo.
//      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
scalar DMI::Energy_Single_Spin( int ispin, const vectorfield & spins )
{
    const auto * geometry            = hamiltonian->geometry.get();
    const auto & boundary_conditions = hamiltonian->boundary_conditions;

    int icell  = ispin / geometry->n_cell_atoms;
    int ibasis = ispin - icell * geometry->n_cell_atoms;

    Pair pair_inv{};
    scalar Energy = 0;
    for( unsigned int ipair = 0; ipair < dmi_pairs.size(); ++ipair )
    {
        const auto & pair = dmi_pairs[ipair];
        if( pair.i == ibasis )
        {
            int jspin = idx_from_pair(
                ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, pair );
            if( jspin >= 0 )
                Energy
                    -= this->dmi_magnitudes[ipair] * this->dmi_normals[ipair].dot( spins[ispin].cross( spins[jspin] ) );
        }
#if !( defined( SPIRIT_USE_OPENMP ) || defined( SPIRIT_USE_CUDA ) )
        if( pair.j == ibasis )
        {
            const auto & t = pair.translations;
            pair_inv       = Pair{ pair.j, pair.i, { -t[0], -t[1], -t[2] } };
            int jspin      = idx_from_pair(
                ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, pair_inv );
            if( jspin >= 0 )
                Energy
                    += this->dmi_magnitudes[ipair] * this->dmi_normals[ipair].dot( spins[ispin].cross( spins[jspin] ) );
        }
#endif
    }
    return Energy;
};

void DMI::Hessian( const vectorfield & spins, MatrixX & hessian )
{
    const auto * geometry            = hamiltonian->geometry.get();
    const auto & boundary_conditions = hamiltonian->boundary_conditions;

#pragma omp parallel for
    for( int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        for( unsigned int i_pair = 0; i_pair < dmi_pairs.size(); ++i_pair )
        {
            int ispin = dmi_pairs[i_pair].i + icell * geometry->n_cell_atoms;
            int jspin = idx_from_pair(
                ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                dmi_pairs[i_pair] );
            if( jspin >= 0 )
            {
                int i = 3 * ispin;
                int j = 3 * jspin;

                hessian( i + 2, j + 1 ) += dmi_magnitudes[i_pair] * dmi_normals[i_pair][0];
                hessian( i + 1, j + 2 ) += -dmi_magnitudes[i_pair] * dmi_normals[i_pair][0];
                hessian( i + 0, j + 2 ) += dmi_magnitudes[i_pair] * dmi_normals[i_pair][1];
                hessian( i + 2, j + 0 ) += -dmi_magnitudes[i_pair] * dmi_normals[i_pair][1];
                hessian( i + 1, j + 0 ) += dmi_magnitudes[i_pair] * dmi_normals[i_pair][2];
                hessian( i + 0, j + 1 ) += -dmi_magnitudes[i_pair] * dmi_normals[i_pair][2];

#if !( defined( SPIRIT_USE_OPENMP ) || defined( SPIRIT_USE_CUDA ) )
                hessian( j + 1, i + 2 ) += dmi_magnitudes[i_pair] * dmi_normals[i_pair][0];
                hessian( j + 2, i + 1 ) += -dmi_magnitudes[i_pair] * dmi_normals[i_pair][0];
                hessian( j + 2, i + 0 ) += dmi_magnitudes[i_pair] * dmi_normals[i_pair][1];
                hessian( j + 0, i + 2 ) += -dmi_magnitudes[i_pair] * dmi_normals[i_pair][1];
                hessian( j + 0, i + 1 ) += dmi_magnitudes[i_pair] * dmi_normals[i_pair][2];
                hessian( j + 1, i + 0 ) += -dmi_magnitudes[i_pair] * dmi_normals[i_pair][2];
#endif
            }
        }
    }
};
void DMI::Sparse_Hessian( const vectorfield & spins, std::vector<triplet> & hessian )
{
    const auto * geometry            = hamiltonian->geometry.get();
    const auto & boundary_conditions = hamiltonian->boundary_conditions;

    for( int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        for( unsigned int i_pair = 0; i_pair < dmi_pairs.size(); ++i_pair )
        {
            int ispin = dmi_pairs[i_pair].i + icell * geometry->n_cell_atoms;
            int jspin = idx_from_pair(
                ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                dmi_pairs[i_pair] );
            if( jspin >= 0 )
            {
                int i = 3 * ispin;
                int j = 3 * jspin;

                hessian.emplace_back( i + 2, j + 1, dmi_magnitudes[i_pair] * dmi_normals[i_pair][0] );
                hessian.emplace_back( i + 1, j + 2, -dmi_magnitudes[i_pair] * dmi_normals[i_pair][0] );
                hessian.emplace_back( i + 0, j + 2, dmi_magnitudes[i_pair] * dmi_normals[i_pair][1] );
                hessian.emplace_back( i + 2, j + 0, -dmi_magnitudes[i_pair] * dmi_normals[i_pair][1] );
                hessian.emplace_back( i + 1, j + 0, dmi_magnitudes[i_pair] * dmi_normals[i_pair][2] );
                hessian.emplace_back( i + 0, j + 1, -dmi_magnitudes[i_pair] * dmi_normals[i_pair][2] );

#if !( defined( SPIRIT_USE_OPENMP ) || defined( SPIRIT_USE_CUDA ) )
                hessian.emplace_back( j + 1, i + 2, dmi_magnitudes[i_pair] * dmi_normals[i_pair][0] );
                hessian.emplace_back( j + 2, i + 1, -dmi_magnitudes[i_pair] * dmi_normals[i_pair][0] );
                hessian.emplace_back( j + 2, i + 0, dmi_magnitudes[i_pair] * dmi_normals[i_pair][1] );
                hessian.emplace_back( j + 0, i + 2, -dmi_magnitudes[i_pair] * dmi_normals[i_pair][1] );
                hessian.emplace_back( j + 0, i + 1, dmi_magnitudes[i_pair] * dmi_normals[i_pair][2] );
                hessian.emplace_back( j + 1, i + 0, -dmi_magnitudes[i_pair] * dmi_normals[i_pair][2] );
#endif
            }
        }
    }
};

#ifdef SPIRIT_USE_CUDA
__global__ void CU_Gradient_DMI(
    const Vector3 * spins, const int * atom_types, const int * boundary_conditions, const int * n_cells,
    int n_cell_atoms, int n_pairs, const Pair * pairs, const scalar * magnitudes, const Vector3 * normals,
    Vector3 * gradient, size_t size )
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
                gradient[ispin] -= magnitudes[ipair] * spins[jspin].cross( normals[ipair] );
            }
        }
    }
}
#endif

void DMI::Gradient( const vectorfield & spins, vectorfield & gradient )
{
    const auto * geometry            = hamiltonian->geometry.get();
    const auto & boundary_conditions = hamiltonian->boundary_conditions;

#ifdef SPIRIT_USE_CUDA
    int size = geometry->n_cells_total;
    CU_Gradient_DMI<<<( size + 1023 ) / 1024, 1024>>>(
        spins.data(), geometry->atom_types.data(), boundary_conditions.data(), geometry->n_cells.data(),
        geometry->n_cell_atoms, this->dmi_pairs.size(), this->dmi_pairs.data(), this->dmi_magnitudes.data(),
        this->dmi_normals.data(), gradient.data(), size );
    CU_CHECK_AND_SYNC();
#else
#pragma omp parallel for
    for( int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        for( unsigned int i_pair = 0; i_pair < dmi_pairs.size(); ++i_pair )
        {
            int ispin = dmi_pairs[i_pair].i + icell * geometry->n_cell_atoms;
            int jspin = idx_from_pair(
                ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                dmi_pairs[i_pair] );
            if( jspin >= 0 )
            {
                gradient[ispin] -= dmi_magnitudes[i_pair] * spins[jspin].cross( dmi_normals[i_pair] );
#ifndef SPIRIT_USE_OPENMP
                gradient[jspin] += dmi_magnitudes[i_pair] * spins[ispin].cross( dmi_normals[i_pair] );
#endif
            }
        }
    }
#endif
};

} // namespace Interaction

} // namespace Engine
