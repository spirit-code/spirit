#include <engine/Backend_par.hpp>
#include <engine/Indexing.hpp>
#include <engine/Neighbours.hpp>
#include <engine/Vectormath.hpp>
#include <engine/interaction/Quadruplet.hpp>
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

Quadruplet::Quadruplet( Hamiltonian * hamiltonian, quadrupletfield quadruplets, scalarfield magnitudes ) noexcept
        : Interaction::Base<Quadruplet>( hamiltonian, scalarfield( 0 ) ),
          quadruplets( std::move( quadruplets ) ),
          quadruplet_magnitudes( std::move( magnitudes ) )
{
    this->updateGeometry();
}

Quadruplet::Quadruplet( Hamiltonian * hamiltonian, const Data::QuadrupletfieldData & quadruplet ) noexcept
        : Quadruplet( hamiltonian, quadruplet.quadruplets, quadruplet.magnitudes )
{
}

bool Quadruplet::is_contributing() const
{
    return !quadruplets.empty();
}

void Quadruplet::updateFromGeometry( const Geometry * geometry ) {}

template<typename Callable>
void Quadruplet::apply( Callable f )
{
    const auto * geometry            = hamiltonian->geometry.get();
    const auto & boundary_conditions = hamiltonian->boundary_conditions;

    for( unsigned int iquad = 0; iquad < quadruplets.size(); ++iquad )
    {
        const auto & quad = quadruplets[iquad];

        const int i = quad.i;
        const int j = quad.j;
        const int k = quad.k;
        const int l = quad.l;

        const auto & d_j = quad.d_j;
        const auto & d_k = quad.d_k;
        const auto & d_l = quad.d_l;

        for( int da = 0; da < geometry->n_cells[0]; ++da )
        {
            for( int db = 0; db < geometry->n_cells[1]; ++db )
            {
                for( int dc = 0; dc < geometry->n_cells[2]; ++dc )
                {
                    int ispin = i + idx_from_translations( geometry->n_cells, geometry->n_cell_atoms, { da, db, dc } );
#ifdef SPIRIT_USE_CUDA
                    int jspin = idx_from_pair(
                        ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                        { i, j, { d_j[0], d_j[1], d_j[2] } } );
                    int kspin = idx_from_pair(
                        ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                        { i, k, { d_k[0], d_k[1], d_k[2] } } );
                    int lspin = idx_from_pair(
                        ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                        { i, l, { d_l[0], d_l[1], d_l[2] } } );
#else
                    int jspin = idx_from_pair(
                        ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                        { i, j, d_j } );
                    int kspin = idx_from_pair(
                        ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                        { i, k, d_k } );
                    int lspin = idx_from_pair(
                        ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                        { i, l, d_l } );
#endif
                    f( iquad, ispin, jspin, kspin, lspin );
                }
            }
        }
    }
}

void Quadruplet::Energy_per_Spin( const vectorfield & spins, scalarfield & energy )
{
    this->apply(
        [&spins, &energy, &quadruplet_magnitudes = this->quadruplet_magnitudes](
            const auto iquad, const auto ispin, const auto jspin, const auto kspin, const auto lspin )
        {
            if( ispin >= 0 && jspin >= 0 && kspin >= 0 && lspin >= 0 )
            {
                const scalar quad_energy = 0.25 * quadruplet_magnitudes[iquad] * ( spins[ispin].dot( spins[jspin] ) )
                                           * ( spins[kspin].dot( spins[lspin] ) );
                energy[ispin] -= quad_energy;
                energy[jspin] -= quad_energy;
                energy[kspin] -= quad_energy;
                energy[lspin] -= quad_energy;
            }
        } );
}

// Calculate the total energy for a single spin to be used in Monte Carlo.
//      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
scalar Quadruplet::Energy_Single_Spin( const int ispin, const vectorfield & spins )
{
    // TODO
    return 0;
};

void Quadruplet::Hessian( const vectorfield & spins, MatrixX & hessian )
{
    // TODO
};
void Quadruplet::Sparse_Hessian( const vectorfield & spins, std::vector<triplet> & hessian )
{
    // TODO
};

void Quadruplet::Gradient( const vectorfield & spins, vectorfield & gradient )
{
    this->apply(
        [&spins, &gradient, &quadruplet_magnitudes = this->quadruplet_magnitudes](
            const auto iquad, const auto ispin, const auto jspin, const auto kspin, const auto lspin )
        {
            if( ispin >= 0 && jspin >= 0 && kspin >= 0 && lspin >= 0 )
            {
                gradient[ispin] -= quadruplet_magnitudes[iquad] * spins[jspin] * ( spins[kspin].dot( spins[lspin] ) );
                gradient[jspin] -= quadruplet_magnitudes[iquad] * spins[ispin] * ( spins[kspin].dot( spins[lspin] ) );
                gradient[kspin] -= quadruplet_magnitudes[iquad] * ( spins[ispin].dot( spins[jspin] ) ) * spins[lspin];
                gradient[lspin] -= quadruplet_magnitudes[iquad] * ( spins[ispin].dot( spins[jspin] ) ) * spins[kspin];
            }
        } );
};

} // namespace Interaction

} // namespace Engine
