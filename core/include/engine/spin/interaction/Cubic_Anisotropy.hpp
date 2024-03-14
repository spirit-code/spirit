#pragma once
#ifndef SPIRIT_CORE_ENGINE_INTERACTION_CUBIC_ANISOTROPY_HPP
#define SPIRIT_CORE_ENGINE_INTERACTION_CUBIC_ANISOTROPY_HPP

#include <engine/Indexing.hpp>
#include <engine/spin/interaction/ABC.hpp>

namespace Engine
{

namespace Spin
{

namespace Interaction
{

struct Cubic_Anisotropy
{
    using state_t = vectorfield;

    struct Data
    {
        intfield cubic_anisotropy_indices;
        scalarfield cubic_anisotropy_magnitudes;
    };

    static bool verify( const Data & data )
    {
        return data.cubic_anisotropy_magnitudes.size() == data.cubic_anisotropy_indices.size();
    };

    struct Cache
    {
    };

    static bool is_contributing( const Data & data, const Cache & )
    {
        return !data.cubic_anisotropy_indices.empty();
    }

    struct IndexType
    {
        int ispin, iani;
    };

    using Index = std::optional<IndexType>;

    static void clearIndex( Index & index )
    {
        index.reset();
    }

    using Energy   = Local::Energy_Functor<Cubic_Anisotropy>;
    using Gradient = Local::Gradient_Functor<Cubic_Anisotropy>;
    using Hessian  = Local::Hessian_Functor<Cubic_Anisotropy>;

    static std::size_t Sparse_Hessian_Size_per_Cell( const Data &, const Cache & )
    {
        return 0;
    };

    // Calculate the total energy for a single spin to be used in Monte Carlo.
    //      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
    using Energy_Single_Spin = Local::Energy_Single_Spin_Functor<Energy, 1>;

    // Interaction name as string
    static constexpr std::string_view name = "Cubic Anisotropy";

    template<typename IndexVector>
    static void applyGeometry(
        const ::Data::Geometry & geometry, const intfield &, const Data & data, Cache &, IndexVector & indices )
    {
        using Indexing::check_atom_type;

#pragma omp parallel for
        for( int icell = 0; icell < geometry.n_cells_total; ++icell )
        {
            for( int iani = 0; iani < data.cubic_anisotropy_indices.size(); ++iani )
            {
                int ispin = icell * geometry.n_cell_atoms + data.cubic_anisotropy_indices[iani];
                if( check_atom_type( geometry.atom_types[ispin] ) )
                    std::get<Index>( indices[ispin] ) = IndexType{ ispin, iani };
            }
        }
    };
};

template<>
template<typename F>
void Cubic_Anisotropy::Hessian::operator()( const Index & index, const vectorfield & spins, F & f ) const
{
    // TODO: Not yet implemented
}

} // namespace Interaction

} // namespace Spin

} // namespace Engine
#endif
