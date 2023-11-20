#include <data/Spin_System.hpp>
#include <engine/Backend_par.hpp>
#include <engine/Indexing.hpp>
#include <engine/Neighbours.hpp>
#include <engine/Vectormath.hpp>
#include <engine/interaction/Anisotropy.hpp>
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

Anisotropy::Anisotropy(
    Hamiltonian * hamiltonian, intfield indices, scalarfield magnitudes, vectorfield normals ) noexcept
        : Interaction::Base<Anisotropy>( hamiltonian, scalarfield( 0 ) ),
          anisotropy_indices( std::move( indices ) ),
          anisotropy_magnitudes( std::move( magnitudes ) ),
          anisotropy_normals( std::move( normals ) )
{
    this->updateGeometry();
}

Anisotropy::Anisotropy( Hamiltonian * hamiltonian, const Data::VectorfieldData & anisotropy ) noexcept
        : Anisotropy( hamiltonian, anisotropy.indices, anisotropy.magnitudes, anisotropy.normals )
{
}

void Anisotropy::updateFromGeometry( const Geometry * geometry ) {}

bool Anisotropy::is_contributing() const
{
    return !anisotropy_indices.empty();
}

#ifdef SPIRIT_USE_CUDA
__global__ void CU_E_Anisotropy(
    const Vector3 * spins, const int * atom_types, const int n_cell_atoms, const int n_anisotropies,
    const int * anisotropy_indices, const scalar * anisotropy_magnitude, const Vector3 * anisotropy_normal,
    scalar * energy, size_t n_cells_total )
{
    for( auto icell = blockIdx.x * blockDim.x + threadIdx.x; icell < n_cells_total; icell += blockDim.x * gridDim.x )
    {
        for( int iani = 0; iani < n_anisotropies; ++iani )
        {
            int ispin = icell * n_cell_atoms + anisotropy_indices[iani];
            if( cu_check_atom_type( atom_types[ispin] ) )
                energy[ispin] -= anisotropy_magnitude[iani] * pow( anisotropy_normal[iani].dot( spins[ispin] ), 2 );
        }
    }
}
#endif

void Anisotropy::Energy_per_Spin( const vectorfield & spins, scalarfield & energy )
{
    const auto * geometry = hamiltonian->geometry.get();
    const int N           = geometry->n_cell_atoms;

#ifdef SPIRIT_USE_CUDA
    int size = geometry->n_cells_total;
    CU_E_Anisotropy<<<( size + 1023 ) / 1024, 1024>>>(
        spins.data(), geometry->atom_types.data(), geometry->n_cell_atoms, this->anisotropy_indices.size(),
        this->anisotropy_indices.data(), this->anisotropy_magnitudes.data(), this->anisotropy_normals.data(),
        energy.data(), size );
    CU_CHECK_AND_SYNC();
#else

#pragma omp parallel for
    for( int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        for( int iani = 0; iani < anisotropy_indices.size(); ++iani )
        {
            int ispin = icell * N + anisotropy_indices[iani];
            if( check_atom_type( geometry->atom_types[ispin] ) )
                energy[ispin] -= this->anisotropy_magnitudes[iani]
                                 * std::pow( anisotropy_normals[iani].dot( spins[ispin] ), 2.0 );
        }
    }
#endif
}

// Calculate the total energy for a single spin to be used in Monte Carlo.
//      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
scalar Anisotropy::Energy_Single_Spin( const int ispin, const vectorfield & spins )
{
    scalar energy         = 0;
    const auto * geometry = hamiltonian->geometry.get();
    const int N           = geometry->n_cell_atoms;

    int icell  = ispin / N;
    int ibasis = ispin - icell * geometry->n_cell_atoms;

    for( int iani = 0; iani < anisotropy_indices.size(); ++iani )
    {
        if( anisotropy_indices[iani] == ibasis )
        {
            if( check_atom_type( geometry->atom_types[ispin] ) )
                energy -= anisotropy_magnitudes[iani] * std::pow( anisotropy_normals[iani].dot( spins[ispin] ), 2.0 );
        }
    }
    return energy;
};

void Anisotropy::Hessian( const vectorfield & spins, MatrixX & hessian )
{
    const auto * geometry = hamiltonian->geometry.get();
    const int N           = geometry->n_cell_atoms;

    // --- Single Spin elements
#pragma omp parallel for
    for( int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        for( int iani = 0; iani < anisotropy_indices.size(); ++iani )
        {
            int ispin = icell * N + anisotropy_indices[iani];
            if( check_atom_type( geometry->atom_types[ispin] ) )
            {
                for( int alpha = 0; alpha < 3; ++alpha )
                {
                    for( int beta = 0; beta < 3; ++beta )
                    {
                        int i = 3 * ispin + alpha;
                        int j = 3 * ispin + alpha;
                        hessian( i, j ) += -2.0 * this->anisotropy_magnitudes[iani]
                                           * this->anisotropy_normals[iani][alpha]
                                           * this->anisotropy_normals[iani][beta];
                    }
                }
            }
        }
    }
};

void Anisotropy::Sparse_Hessian( const vectorfield & spins, std::vector<triplet> & hessian )
{
    const auto * geometry = hamiltonian->geometry.get();
    const int N           = geometry->n_cell_atoms;

    // --- Single Spin elements
    for( int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        for( int iani = 0; iani < anisotropy_indices.size(); ++iani )
        {
            int ispin = icell * N + anisotropy_indices[iani];
            if( check_atom_type( geometry->atom_types[ispin] ) )
            {
                for( int alpha = 0; alpha < 3; ++alpha )
                {
                    for( int beta = 0; beta < 3; ++beta )
                    {
                        int i      = 3 * ispin + alpha;
                        int j      = 3 * ispin + alpha;
                        scalar res = -2.0 * this->anisotropy_magnitudes[iani] * this->anisotropy_normals[iani][alpha]
                                     * this->anisotropy_normals[iani][beta];
                        hessian.emplace_back( i, j, res );
                    }
                }
            }
        }
    }
};

#ifdef SPIRIT_USE_CUDA
__global__ void CU_Gradient_Anisotropy(
    const Vector3 * spins, const int * atom_types, const int n_cell_atoms, const int n_anisotropies,
    const int * anisotropy_indices, const scalar * anisotropy_magnitude, const Vector3 * anisotropy_normal,
    Vector3 * gradient, size_t n_cells_total )
{
    for( auto icell = blockIdx.x * blockDim.x + threadIdx.x; icell < n_cells_total; icell += blockDim.x * gridDim.x )
    {
        for( int iani = 0; iani < n_anisotropies; ++iani )
        {
            int ispin = icell * n_cell_atoms + anisotropy_indices[iani];
            if( cu_check_atom_type( atom_types[ispin] ) )
            {
                scalar sc = -2 * anisotropy_magnitude[iani] * anisotropy_normal[iani].dot( spins[ispin] );
                gradient[ispin] += sc * anisotropy_normal[iani];
            }
        }
    }
}
#endif

void Anisotropy::Gradient( const vectorfield & spins, vectorfield & gradient )
{
    const auto * geometry = hamiltonian->geometry.get();
    const int N           = geometry->n_cell_atoms;

#ifdef SPIRIT_USE_CUDA
    int size = geometry->n_cells_total;
    CU_Gradient_Anisotropy<<<( size + 1023 ) / 1024, 1024>>>(
        spins.data(), geometry->atom_types.data(), geometry->n_cell_atoms, this->anisotropy_indices.size(),
        this->anisotropy_indices.data(), this->anisotropy_magnitudes.data(), this->anisotropy_normals.data(),
        gradient.data(), size );
    CU_CHECK_AND_SYNC();
#else

#pragma omp parallel for
    for( int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        for( int iani = 0; iani < anisotropy_indices.size(); ++iani )
        {
            int ispin = icell * N + anisotropy_indices[iani];
            if( check_atom_type( geometry->atom_types[ispin] ) )
                gradient[ispin] -= 2.0 * this->anisotropy_magnitudes[iani] * this->anisotropy_normals[iani]
                                   * anisotropy_normals[iani].dot( spins[ispin] );
        }
    }
#endif
};

} // namespace Interaction

} // namespace Engine
