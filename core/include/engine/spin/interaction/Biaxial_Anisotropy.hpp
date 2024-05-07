#pragma once
#ifndef SPIRIT_CORE_ENGINE_INTERACTION_BIAXIAL_ANISOTROPY_HPP
#define SPIRIT_CORE_ENGINE_INTERACTION_BIAXIAL_ANISOTROPY_HPP

#include <engine/Indexing.hpp>
#include <engine/spin/interaction/Functor_Prototypes.hpp>

#include <Eigen/Dense>

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

    using Index        = const IndexType *;
    using IndexStorage = Backend::optional<IndexType>;

    using Energy   = Functor::Local::Energy_Functor<Functor::Local::DataRef<Biaxial_Anisotropy>>;
    using Gradient = Functor::Local::Gradient_Functor<Functor::Local::DataRef<Biaxial_Anisotropy>>;
    using Hessian  = Functor::Local::Hessian_Functor<Functor::Local::DataRef<Biaxial_Anisotropy>>;

    static std::size_t Sparse_Hessian_Size_per_Cell( const Data & data, const Cache & )
    {
        return data.indices.size() * 9;
    };

    // Calculate the total energy for a single spin to be used in Monte Carlo.
    //      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
    using Energy_Single_Spin = Functor::Local::Energy_Single_Spin_Functor<Energy, 1>;

    // Interaction name as string
    static constexpr std::string_view name = "Biaxial Anisotropy";

    template<typename IndexStorageVector>
    static void applyGeometry(
        const ::Data::Geometry & geometry, const intfield &, const Data & data, Cache &, IndexStorageVector & indices )
    {
        using Indexing::check_atom_type;
        const auto N = geometry.n_cell_atoms;

        for( int icell = 0; icell < geometry.n_cells_total; ++icell )
        {
            for( int iani = 0; iani < data.indices.size(); ++iani )
            {
                int ispin = icell * N + data.indices[iani];
                if( check_atom_type( geometry.atom_types[ispin] ) )
                    Backend::get<IndexStorage>( indices[ispin] ) = IndexType{ ispin, iani };
            }
        }
    };
};

template<>
struct Functor::Local::DataRef<Biaxial_Anisotropy>
{
    using Interaction = Biaxial_Anisotropy;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;

    DataRef( const Data & data, const Cache & cache ) noexcept
            : is_contributing( Interaction::is_contributing( data, cache ) ),
              bases( data.bases.data() ),
              site_p( data.site_p.data() ),
              terms( data.terms.data() ){};

    const bool is_contributing;

protected:
    const PolynomialBasis * bases;
    const unsigned int * site_p;
    const PolynomialTerm * terms;
};

template<>
inline scalar Biaxial_Anisotropy::Energy::operator()( const Index & index, const Vector3 * spins ) const
{
    using std::pow;
    scalar result = 0;
    if( !is_contributing || index == nullptr )
        return result;

    const auto & [ispin, iani] = *index;
    const scalar s1            = bases[iani].k1.dot( spins[ispin] );
    const scalar s2            = bases[iani].k2.dot( spins[ispin] );
    const scalar s3            = bases[iani].k3.dot( spins[ispin] );

    const scalar sin_theta_2 = 1 - s1 * s1;

    for( auto iterm = site_p[iani]; iterm < site_p[iani + 1]; ++iterm )
    {
        const auto & [coeff, n1, n2, n3] = terms[iterm];
        result += coeff * pow( sin_theta_2, n1 ) * pow( s2, n2 ) * pow( s3, n3 );
    }

    return result;
}

template<>
inline Vector3 Biaxial_Anisotropy::Gradient::operator()( const Index & index, const Vector3 * spins ) const
{
    using std::pow;
    Vector3 result = Vector3::Zero();
    if( !is_contributing || index == nullptr )
        return result;

    const auto & [ispin, iani] = *index;
    const auto & [k1, k2, k3]  = bases[iani];

    const scalar s1 = k1.dot( spins[ispin] );
    const scalar s2 = k2.dot( spins[ispin] );
    const scalar s3 = k3.dot( spins[ispin] );

    const scalar sin_theta_2 = 1 - s1 * s1;

    for( auto iterm = site_p[iani]; iterm < site_p[iani + 1]; ++iterm )
    {
        const auto & [coeff, n1, n2, n3] = terms[iterm];

        const scalar a = pow( s2, n2 );
        const scalar b = pow( s3, n3 );
        const scalar c = pow( sin_theta_2, n1 );

        if( n1 > 0 )
            result += k1 * ( coeff * a * b * n1 * ( -2.0 * s1 * pow( sin_theta_2, n1 - 1 ) ) );
        if( n2 > 0 )
            result += k2 * ( coeff * b * c * n2 * pow( s2, n2 - 1 ) );
        if( n3 > 0 )
            result += k3 * ( coeff * a * c * n3 * pow( s3, n3 - 1 ) );
    }

    return result;
}

template<>
template<typename Callable>
void Biaxial_Anisotropy::Hessian::operator()( const Index & index, const vectorfield & spins, Callable & hessian ) const
{
    using std::pow;
    if( !is_contributing || index == nullptr )
        return;

    const auto & [ispin, iani] = *index;
    const auto & [k1, k2, k3]  = bases[iani];

    const scalar s1 = k1.dot( spins[ispin] );
    const scalar s2 = k2.dot( spins[ispin] );
    const scalar s3 = k3.dot( spins[ispin] );

    const scalar st2 = 1 - s1 * s1;

    for( auto iterm = site_p[iani]; iterm < site_p[iani + 1]; ++iterm )
    {
        const auto & [coeff, n1, n2, n3] = terms[iterm];

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
            : b * coeff * n2 * pow( s2, n2 - 1 ) * ( -2.0 * n1 * s1 ) * pow( s1, n1 - 1 );
        const scalar p_13 = n3 == 0 || n1 == 0 ? 0
            : a * coeff * n3 * pow( s3, n3 - 1 ) * ( -2.0 * n1 * s1 ) * pow( s1, n1 - 1 );
        const scalar p_23 = n2 == 0 || n3 == 0 ? 0
            : c * coeff * n2 * pow( s2, n2 - 1 ) * n3 * pow( s3, n3 - 1 );
        // clang-format on

#pragma unroll
        for( int alpha = 0; alpha < 3; ++alpha )
        {
#pragma unroll
            for( int beta = 0; beta < 3; ++beta )
            {
                hessian(
                    3 * ispin + alpha, 3 * ispin + beta,
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
