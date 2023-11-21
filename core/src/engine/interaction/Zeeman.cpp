#include <engine/Backend_par.hpp>
#include <engine/Indexing.hpp>
#include <engine/Neighbours.hpp>
#include <engine/Vectormath.hpp>
#include <engine/interaction/Zeeman.hpp>
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

Zeeman::Zeeman( Hamiltonian * hamiltonian, scalar magnitude, Vector3 normal ) noexcept
        : Interaction::Base<Zeeman>( hamiltonian, scalarfield( 0 ) ),
          external_field_magnitude( magnitude ),
          external_field_normal( std::move( normal ) )
{
    this->updateGeometry();
}

Zeeman::Zeeman( Hamiltonian * hamiltonian, const Data::NormalVector & external_field ) noexcept
        : Zeeman( hamiltonian, external_field.magnitude, external_field.normal )
{
}

bool Zeeman::is_contributing() const
{
    return std::abs( this->external_field_magnitude ) > 1e-60;
}

void Zeeman::updateFromGeometry( const Geometry * geometry ){};

#ifdef SPIRIT_USE_CUDA
__global__ void CU_E_Zeeman(
    const Vector3 * spins, const int * atom_types, const int n_cell_atoms, const scalar * mu_s,
    const scalar external_field_magnitude, const Vector3 external_field_normal, scalar * energy, size_t n_cells_total )
{
    for( auto icell = blockIdx.x * blockDim.x + threadIdx.x; icell < n_cells_total; icell += blockDim.x * gridDim.x )
    {
        for( int ibasis = 0; ibasis < n_cell_atoms; ++ibasis )
        {
            int ispin = n_cell_atoms * icell + ibasis;
            if( cu_check_atom_type( atom_types[ispin] ) )
                energy[ispin] -= mu_s[ispin] * external_field_magnitude * external_field_normal.dot( spins[ispin] );
        }
    }
}
#endif

void Zeeman::Energy_per_Spin( const vectorfield & spins, scalarfield & energy )
{
    const auto * geometry = hamiltonian->geometry.get();
    const auto N          = geometry->n_cell_atoms;
    const auto & mu_s     = geometry->mu_s;

#ifdef SPIRIT_USE_CUDA
    int size = geometry->n_cells_total;
    CU_E_Zeeman<<<( size + 1023 ) / 1024, 1024>>>(
        spins.data(), geometry->atom_types.data(), N, geometry->mu_s.data(), this->external_field_magnitude,
        this->external_field_normal, energy.data(), size );
    CU_CHECK_AND_SYNC();
#else

#pragma omp parallel for
    for( int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        for( int ibasis = 0; ibasis < N; ++ibasis )
        {
            int ispin = icell * N + ibasis;
            if( check_atom_type( geometry->atom_types[ispin] ) )
                energy[ispin]
                    -= mu_s[ispin] * this->external_field_magnitude * this->external_field_normal.dot( spins[ispin] );
        }
    }
#endif
}

// Calculate the total energy for a single spin to be used in Monte Carlo.
//      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
scalar Zeeman::Energy_Single_Spin( const int ispin, const vectorfield & spins )
{
    const auto * geometry = hamiltonian->geometry.get();
    const auto & mu_s     = geometry->mu_s;
    return -mu_s[ispin] * this->external_field_magnitude * this->external_field_normal.dot( spins[ispin] );
};

void Zeeman::Hessian( const vectorfield & spins, MatrixX & hessian ){};
void Zeeman::Sparse_Hessian( const vectorfield & spins, std::vector<triplet> & hessian ){};

#ifdef SPIRIT_USE_CUDA
__global__ void CU_Gradient_Zeeman(
    const int * atom_types, const int n_cell_atoms, const scalar * mu_s, const scalar external_field_magnitude,
    const Vector3 external_field_normal, Vector3 * gradient, size_t n_cells_total )
{
    for( auto icell = blockIdx.x * blockDim.x + threadIdx.x; icell < n_cells_total; icell += blockDim.x * gridDim.x )
    {
        for( int ibasis = 0; ibasis < n_cell_atoms; ++ibasis )
        {
            int ispin = n_cell_atoms * icell + ibasis;
            if( cu_check_atom_type( atom_types[ispin] ) )
                gradient[ispin] -= mu_s[ispin] * external_field_magnitude * external_field_normal;
        }
    }
}
#endif

void Zeeman::Gradient( const vectorfield & spins, vectorfield & gradient )
{
    const auto * geometry = hamiltonian->geometry.get();
    const auto N          = geometry->n_cell_atoms;
    const auto & mu_s     = geometry->mu_s;

#ifdef SPIRIT_USE_CUDA
    int size = geometry->n_cells_total;
    CU_Gradient_Zeeman<<<( size + 1023 ) / 1024, 1024>>>(
        geometry->atom_types.data(), geometry->n_cell_atoms, geometry->mu_s.data(), this->external_field_magnitude,
        this->external_field_normal, gradient.data(), size );
    CU_CHECK_AND_SYNC();
#else

#pragma omp parallel for
    for( int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        for( int ibasis = 0; ibasis < N; ++ibasis )
        {
            int ispin = icell * N + ibasis;
            if( check_atom_type( geometry->atom_types[ispin] ) )
                gradient[ispin] -= mu_s[ispin] * this->external_field_magnitude * this->external_field_normal;
        }
    }
#endif
};

} // namespace Interaction

} // namespace Engine
