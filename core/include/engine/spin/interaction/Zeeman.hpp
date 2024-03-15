#pragma once
#ifndef SPIRIT_CORE_ENGINE_INTERACTION_ZEEMANN_HPP
#define SPIRIT_CORE_ENGINE_INTERACTION_ZEEMANN_HPP

#include <engine/Indexing.hpp>
#include <engine/spin/interaction/Functor_Prototpyes.hpp>

#include <optional>

namespace Engine
{

namespace Spin
{

namespace Interaction
{

struct Zeeman
{
    using state_t = vectorfield;

    struct Data
    {
        scalar external_field_magnitude = 0;
        Vector3 external_field_normal   = { 0, 0, 1 };

        Data( scalar external_field_magnitude, Vector3 external_field_normal )
                : external_field_magnitude( external_field_magnitude ),
                  external_field_normal( std::move( external_field_normal ) ){};
    };

    // clang-tidy: ignore
    typedef int IndexType;

    using Index = std::optional<IndexType>;

    struct Cache
    {
        const ::Data::Geometry * geometry;
    };

    static bool is_contributing( const Data & data, const Cache & )
    {
        return std::abs( data.external_field_magnitude ) > 1e-60;
    }

    using Energy   = Functor::Local::Energy_Functor<Zeeman>;
    using Gradient = Functor::Local::Gradient_Functor<Zeeman>;
    using Hessian  = Functor::Local::Hessian_Functor<Zeeman>;

    static std::size_t Sparse_Hessian_Size_per_Cell( const Data &, const Cache & )
    {
        return 0;
    };

    // Calculate the total energy for a single spin to be used in Monte Carlo.
    //      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
    using Energy_Single_Spin = Functor::Local::Energy_Single_Spin_Functor<Energy, 1>;

    // Interaction name as string
    static constexpr std::string_view name = "Zeeman";

    template<typename IndexVector>
    static void applyGeometry(
        const ::Data::Geometry & geometry, const intfield &, const Data &, Cache & cache, IndexVector & indices )
    {
        using Indexing::check_atom_type;

        const auto N = geometry.n_cell_atoms;

        for( int icell = 0; icell < geometry.n_cells_total; ++icell )
        {
            for( int ibasis = 0; ibasis < N; ++ibasis )
            {
                const int ispin = icell * N + ibasis;
                if( check_atom_type( geometry.atom_types[ispin] ) )
                {
                    std::get<Index>( indices[ispin] ) = ispin;
                }
            };
        }

        cache.geometry = &geometry;
    }
};

template<>
inline scalar Zeeman::Energy::operator()( const Index & index, const vectorfield & spins ) const
{
    if( cache.geometry == nullptr )
        return 0;

    if( index.has_value() && *index >= 0 )
    {
        const auto & mu_s  = cache.geometry->mu_s;
        const auto & ispin = *index;
        return -mu_s[ispin] * data.external_field_magnitude * data.external_field_normal.dot( spins[ispin] );
    }
    else
        return 0;
}

template<>
inline Vector3 Zeeman::Gradient::operator()( const Index & index, const vectorfield & ) const
{
    if( cache.geometry == nullptr )
        return Vector3::Zero();

    if( index.has_value() && *index >= 0 )
    {
        const auto & mu_s  = cache.geometry->mu_s;
        const auto & ispin = *index;
        return -mu_s[ispin] * data.external_field_magnitude * data.external_field_normal;
    }
    else
        return Vector3::Zero();
}

template<>
template<typename F>
void Zeeman::Hessian::operator()( const Index &, const vectorfield &, F & ) const {};

} // namespace Interaction

} // namespace Spin

} // namespace Engine
#endif
