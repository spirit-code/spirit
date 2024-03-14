#pragma once
#ifndef SPIRIT_CORE_ENGINE_INTERACTION_BIAXIAL_ANISOTROPY_HPP
#define SPIRIT_CORE_ENGINE_INTERACTION_BIAXIAL_ANISOTROPY_HPP

#include <engine/Indexing.hpp>
#include <engine/spin/interaction/ABC.hpp>

namespace Engine
{

namespace Spin
{

namespace Interaction
{

/*
 * Biaxial Anisotropy
 * The terms use a CSR like format. The site_p attribute stores the information which term corresponds to which site,
 * such that the terms for the atom at `indices[i]` are the ones between `site_p[i]` & `site_p[i+1]`.
 */
struct Biaxial_Anisotropy
{
    using state_t = vectorfield;

    struct Data
    {
        intfield indices;
        field<PolynomialBasis> bases;
        field<unsigned int> site_p;
        field<PolynomialTerm> terms;

        Data( intfield indices, field<PolynomialBasis> bases, field<unsigned int> site_p, field<PolynomialTerm> terms )
                : indices( std::move( indices ) ),
                  bases( std::move( bases ) ),
                  site_p( std::move( site_p ) ),
                  terms( std::move( terms ) ){};
    };

    static bool valid_data( const Data & data )
    {
        if( data.indices.size() != data.bases.size() )
            return false;
        if( ( !data.indices.empty() || !data.site_p.empty() ) && ( data.indices.size() + 1 != data.site_p.size() ) )
            return false;
        if( !data.site_p.empty() && data.site_p.back() != data.terms.size() )
            return false;

        return true;
    }

    struct Cache
    {
    };

    static bool is_contributing( const Data & data, const Cache & )
    {
        return !data.indices.empty();
    };

    struct IndexType
    {
        int ispin, iani;
    };

    using Index = std::optional<IndexType>;

    static void clearIndex( Index & index )
    {
        index.reset();
    }

    using Energy   = Local::Energy_Functor<Biaxial_Anisotropy>;
    using Gradient = Local::Gradient_Functor<Biaxial_Anisotropy>;
    using Hessian  = Local::Hessian_Functor<Biaxial_Anisotropy>;

    static std::size_t Sparse_Hessian_Size_per_Cell( const Data & data, const Cache & )
    {
        return data.indices.size() * 9;
    };

    // Calculate the total energy for a single spin to be used in Monte Carlo.
    //      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
    using Energy_Single_Spin = Local::Energy_Single_Spin_Functor<Energy, 1>;

    // Interaction name as string
    static constexpr std::string_view name = "Biaxial Anisotropy";

    template<typename IndexVector>
    static void applyGeometry(
        const ::Data::Geometry & geometry, const intfield &, const Data & data, Cache &, IndexVector & indices )
    {
        using Indexing::check_atom_type;
        const auto N = geometry.n_cell_atoms;

#pragma omp parallel for
        for( int icell = 0; icell < geometry.n_cells_total; ++icell )
        {
            for( int iani = 0; iani < data.indices.size(); ++iani )
            {
                int ispin = icell * N + data.indices[iani];
                if( check_atom_type( geometry.atom_types[ispin] ) )

                    std::get<Index>( indices[ispin] ) = IndexType{ ispin, iani };
            }
        }
    };
};

template<>
template<typename F>
void Biaxial_Anisotropy::Hessian::operator()( const Index & index, const vectorfield & spins, F & f ) const
{
    using std::pow;
    if( !index.has_value() )
        return;

    const auto & [ispin, iani] = *index;
    const auto & [k1, k2, k3]  = data.bases[iani];

    const scalar s1 = k1.dot( spins[ispin] );
    const scalar s2 = k2.dot( spins[ispin] );
    const scalar s3 = k3.dot( spins[ispin] );

    const scalar st2 = 1 - s1 * s1;

    for( auto iterm = data.site_p[iani]; iterm < data.site_p[iani + 1]; ++iterm )
    {
        const auto & [coeff, n1, n2, n3] = data.terms[iterm];

        const scalar a = pow( s2, n2 );
        const scalar b = pow( s3, n3 );
        const scalar c = pow( st2, n1 );
        // clang-format off
        const scalar p_11 = n1 <= 1 ? 0
            : 2 * n1 * ( 2 * n1 * s1 * s1 - 1 ) * ( coeff * a * b * pow( st2, n1 - 2 ) );
        const scalar p_22 = n2 <= 1 ? 0
            : n2 * ( n2 - 1 ) * ( coeff * b * c * pow( s2, n2 - 2 ) );
        const scalar p_33 = n3 <= 1 ? 0
            : n3 * ( n3 - 1 ) * ( coeff * a * c * pow( s3, n3 - 2 ) );
        const scalar p_12 = n2 == 0 || n1 == 0 ? 0
            : b * coeff * n2 * pow( s2, n2 - 1 ) * ( -2 * n1 * s1 ) * pow( s1, n1 - 1 );
        const scalar p_13 = n3 == 0 || n1 == 0 ? 0
            : a * coeff * n3 * pow( s3, n3 - 1 ) * ( -2 * n1 * s1 ) * pow( s1, n1 - 1 );
        const scalar p_23 = n2 == 0 || n3 == 0 ? 0
            : c * coeff * n2 * pow( s2, n2 - 1 ) * n3 * pow( s3, n3 - 1 );
        // clang-format on

#pragma unroll
        for( int alpha = 0; alpha < 3; ++alpha )
        {
#pragma unroll
            for( int beta = 0; beta < 3; ++beta )
            {
                f( 3 * ispin + alpha, 3 * ispin + beta,
                   k1[alpha] * ( p_11 * k1[beta] + p_12 * k2[beta] + p_13 * k3[beta] )
                       + k2[alpha] * ( p_12 * k1[beta] + p_22 * k2[beta] + p_23 * k3[beta] )
                       + k3[alpha] * ( p_13 * k1[beta] + p_23 * k2[beta] + p_33 * k3[beta] ) );
            }
        }
    }
}

} // namespace Interaction

} // namespace Spin

} // namespace Engine
#endif
