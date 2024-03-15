#pragma once
#ifndef SPIRIT_CORE_ENGINE_INTERACTION_QUADRUPLET_HPP
#define SPIRIT_CORE_ENGINE_INTERACTION_QUADRUPLET_HPP

#include <engine/Indexing.hpp>
#include <engine/spin/interaction/Functor_Prototpyes.hpp>

namespace Engine
{

namespace Spin
{

namespace Interaction
{

struct Quadruplet
{
    using state_t = vectorfield;

    struct Data
    {
        quadrupletfield quadruplets;
        scalarfield magnitudes;

        Data( quadrupletfield quadruplets, scalarfield magnitudes )
                : quadruplets( std::move( quadruplets ) ), magnitudes( std::move( magnitudes ) ){};
    };

    static bool valid_data( const Data & data )
    {
        return data.quadruplets.size() == data.magnitudes.size();
    };

    struct Cache
    {
        const ::Data::Geometry * geometry{};
        const intfield * boundary_conditions{};
    };

    static bool is_contributing( const Data & data, const Cache & )
    {
        return !data.quadruplets.empty();
    }

    using Energy       = Functor::NonLocal::Energy_Functor<Quadruplet>;
    using Gradient     = Functor::NonLocal::Gradient_Functor<Quadruplet>;
    using Hessian      = Functor::NonLocal::Hessian_Functor<Quadruplet>;
    using Energy_Total = Functor::NonLocal::Reduce_Functor<Energy>;

    static std::size_t Sparse_Hessian_Size_per_Cell( const Data &, const Cache & )
    {
        return 0;
    };

    // Calculate the total energy for a single spin to be used in Monte Carlo.
    //      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
    static constexpr scalar weight = 4.0;
    using Energy_Single_Spin       = Functor::NonLocal::Energy_Single_Spin_Functor<Quadruplet>;

    // Interaction name as string
    static constexpr std::string_view name = "Quadruplet";

    static constexpr bool local = false;

    static void applyGeometry(
        const ::Data::Geometry & geometry, const intfield & boundary_conditions, const Data &, Cache & cache )
    {
        cache.geometry            = &geometry;
        cache.boundary_conditions = &boundary_conditions;
    };

    friend void Energy::operator()( const vectorfield & spins, scalarfield & energy ) const;
    friend void Gradient::operator()( const vectorfield & spins, vectorfield & gradient ) const;

private:
    template<typename Callable>
    static void apply( Callable && f, const Data & data, const Cache & cache )
    {
        if( !cache.geometry || !cache.boundary_conditions )
            // TODO: turn this into an error
            return;

        const auto & geometry            = *cache.geometry;
        const auto & boundary_conditions = *cache.boundary_conditions;
        using Indexing::idx_from_pair;

        for( int iquad = 0; iquad < data.quadruplets.size(); ++iquad )
        {
            const auto & quad = data.quadruplets[iquad];

            const int i = quad.i;
            const int j = quad.j;
            const int k = quad.k;
            const int l = quad.l;

            const auto & d_j = quad.d_j;
            const auto & d_k = quad.d_k;
            const auto & d_l = quad.d_l;

            for( unsigned int icell = 0; icell < geometry.n_cells_total; ++icell )
            {
                int ispin = i + icell * geometry.n_cell_atoms;
#ifdef SPIRIT_USE_CUDA
                int jspin = idx_from_pair(
                    ispin, boundary_conditions, geometry.n_cells, geometry.n_cell_atoms, geometry.atom_types,
                    { i, j, { d_j[0], d_j[1], d_j[2] } } );
                int kspin = idx_from_pair(
                    ispin, boundary_conditions, geometry.n_cells, geometry.n_cell_atoms, geometry.atom_types,
                    { i, k, { d_k[0], d_k[1], d_k[2] } } );
                int lspin = idx_from_pair(
                    ispin, boundary_conditions, geometry.n_cells, geometry.n_cell_atoms, geometry.atom_types,
                    { i, l, { d_l[0], d_l[1], d_l[2] } } );
#else
                int jspin = idx_from_pair(
                    ispin, boundary_conditions, geometry.n_cells, geometry.n_cell_atoms, geometry.atom_types,
                    { i, j, d_j } );
                int kspin = idx_from_pair(
                    ispin, boundary_conditions, geometry.n_cells, geometry.n_cell_atoms, geometry.atom_types,
                    { i, k, d_k } );
                int lspin = idx_from_pair(
                    ispin, boundary_conditions, geometry.n_cells, geometry.n_cell_atoms, geometry.atom_types,
                    { i, l, d_l } );
#endif
                f( iquad, ispin, jspin, kspin, lspin );
            }
        }
    }
};

template<>
template<typename F>
void Quadruplet::Hessian::operator()( const vectorfield & spins, F & f ) const {
    // TODO
};

} // namespace Interaction

} // namespace Spin

} // namespace Engine

#endif
