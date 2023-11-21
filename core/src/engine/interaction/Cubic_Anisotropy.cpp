#include <engine/Backend_par.hpp>
#include <engine/Indexing.hpp>
#include <engine/Neighbours.hpp>
#include <engine/Vectormath.hpp>
#include <engine/interaction/Cubic_Anisotropy.hpp>
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

Cubic_Anisotropy::Cubic_Anisotropy( Hamiltonian * hamiltonian, intfield indices, scalarfield magnitudes ) noexcept
        : Interaction::Base<Cubic_Anisotropy>( hamiltonian, scalarfield( 0 ) ),
          cubic_anisotropy_indices( std::move( indices ) ),
          cubic_anisotropy_magnitudes( std::move( magnitudes ) )
{
    this->updateGeometry();
}

Cubic_Anisotropy::Cubic_Anisotropy( Hamiltonian * hamiltonian, const Data::ScalarfieldData & cubic_anisotropy ) noexcept
        : Cubic_Anisotropy( hamiltonian, cubic_anisotropy.indices, cubic_anisotropy.magnitudes )
{
}

bool Cubic_Anisotropy::is_contributing() const
{
    return !cubic_anisotropy_indices.empty();
}

void Cubic_Anisotropy::updateFromGeometry( const Geometry * geometry ){};

#ifdef SPIRIT_USE_CUDA
__global__ void CU_E_Cubic_Anisotropy(
    const Vector3 * spins, const int * atom_types, const int n_cell_atoms, const int n_anisotropies,
    const int * anisotropy_indices, const scalar * anisotropy_magnitude, scalar * energy, size_t n_cells_total )
{
    for( auto icell = blockIdx.x * blockDim.x + threadIdx.x; icell < n_cells_total; icell += blockDim.x * gridDim.x )
    {
        for( int iani = 0; iani < n_anisotropies; ++iani )
        {
            int ispin = icell * n_cell_atoms + anisotropy_indices[iani];
            if( cu_check_atom_type( atom_types[ispin] ) )
                energy[ispin]
                    -= 0.5 * anisotropy_magnitude[iani]
                       * ( pow( spins[ispin][0], 4.0 ) + pow( spins[ispin][1], 4.0 ) + pow( spins[ispin][2], 4.0 ) );
        }
    }
}
#endif

void Cubic_Anisotropy::Energy_per_Spin( const vectorfield & spins, scalarfield & energy )
{
    const auto * geometry = hamiltonian->geometry.get();
    const auto N          = geometry->n_cell_atoms;

#ifdef SPIRIT_USE_CUDA
    int size = geometry->n_cells_total;
    CU_E_Cubic_Anisotropy<<<( size + 1023 ) / 1024, 1024>>>(
        spins.data(), geometry->atom_types.data(), N, this->cubic_anisotropy_indices.size(),
        this->cubic_anisotropy_indices.data(), this->cubic_anisotropy_magnitudes.data(), energy.data(), size );
    CU_CHECK_AND_SYNC();
#else

#pragma omp parallel for
    for( int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        for( int iani = 0; iani < cubic_anisotropy_indices.size(); ++iani )
        {
            int ispin = icell * N + cubic_anisotropy_indices[iani];
            if( check_atom_type( geometry->atom_types[ispin] ) )
                energy[ispin] -= 0.5 * this->cubic_anisotropy_magnitudes[iani]
                                 * ( std::pow( spins[ispin][0], 4.0 ) + std::pow( spins[ispin][1], 4.0 )
                                     + std::pow( spins[ispin][2], 4.0 ) );
        }
    }
#endif
};

// Calculate the total energy for a single spin to be used in Monte Carlo.
//      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
scalar Cubic_Anisotropy::Energy_Single_Spin( const int ispin, const vectorfield & spins )
{
    const auto * geometry = hamiltonian->geometry.get();

    int icell  = ispin / geometry->n_cell_atoms;
    int ibasis = ispin - icell * geometry->n_cell_atoms;

    scalar energy = 0;
    for( int iani = 0; iani < cubic_anisotropy_indices.size(); ++iani )
    {
        if( cubic_anisotropy_indices[iani] == ibasis )
        {
            if( check_atom_type( geometry->atom_types[ispin] ) )
                energy -= 0.5 * this->cubic_anisotropy_magnitudes[iani]
                          * ( std::pow( spins[ispin][0], 4.0 ) + std::pow( spins[ispin][1], 4.0 )
                              + std::pow( spins[ispin][2], 4.0 ) );
        }
    }
    return energy;
};

void Cubic_Anisotropy::Hessian( const vectorfield & spins, MatrixX & hessian )
{
    // TODO
};

void Cubic_Anisotropy::Sparse_Hessian( const vectorfield & spins, std::vector<triplet> & hessian )
{
    // TODO
};

#ifdef SPIRIT_USE_CUDA
__global__ void CU_Gradient_Cubic_Anisotropy(
    const Vector3 * spins, const int * atom_types, const int n_cell_atoms, const int n_anisotropies,
    const int * anisotropy_indices, const scalar * anisotropy_magnitude, Vector3 * gradient, size_t n_cells_total )
{
    for( auto icell = blockIdx.x * blockDim.x + threadIdx.x; icell < n_cells_total; icell += blockDim.x * gridDim.x )
    {
        for( int iani = 0; iani < n_anisotropies; ++iani )
        {
            int ispin = icell * n_cell_atoms + anisotropy_indices[iani];
            if( cu_check_atom_type( atom_types[ispin] ) )
            {
                for( int icomp = 0; icomp < 3; ++icomp )
                {
                    gradient[ispin][icomp] -= 2.0 * anisotropy_magnitude[iani] * pow( spins[ispin][icomp], 3.0 );
                }
            }
        }
    }
}
#endif

void Cubic_Anisotropy::Gradient( const vectorfield & spins, vectorfield & gradient )
{
    const auto * geometry = hamiltonian->geometry.get();
    const auto N          = geometry->n_cell_atoms;

#ifdef SPIRIT_USE_CUDA
    int size = geometry->n_cells_total;
    CU_Gradient_Cubic_Anisotropy<<<( size + 1023 ) / 1024, 1024>>>(
        spins.data(), geometry->atom_types.data(), geometry->n_cell_atoms, this->cubic_anisotropy_indices.size(),
        this->cubic_anisotropy_indices.data(), this->cubic_anisotropy_magnitudes.data(), gradient.data(), size );
    CU_CHECK_AND_SYNC();
#else
#pragma omp parallel for
    for( int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        for( int iani = 0; iani < cubic_anisotropy_indices.size(); ++iani )
        {
            int ispin = icell * N + cubic_anisotropy_indices[iani];
            if( check_atom_type( geometry->atom_types[ispin] ) )
                for( int icomp = 0; icomp < 3; ++icomp )
                {
                    gradient[ispin][icomp]
                        -= 2.0 * this->cubic_anisotropy_magnitudes[iani] * std::pow( spins[ispin][icomp], 3.0 );
                }
        }
    }
#endif
};

} // namespace Interaction

} // namespace Engine
