#pragma once
#ifndef SPIRIT_CORE_ENGINE_INTERACTION_BIAXIAL_ANISOTROPY_HPP
#define SPIRIT_CORE_ENGINE_INTERACTION_BIAXIAL_ANISOTROPY_HPP

#include <engine/Index_Container.hpp>
#include <engine/Indexing.hpp>
#include <engine/spin/StateType.hpp>
#include <engine/spin/interaction/Functor_Prototypes.hpp>
#include <utility/Fastpow.hpp>

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
    using state_t = StateType;

    struct Data
    {
        intfield indices{};
        field<PolynomialBasis> bases{};
        field<unsigned int> site_p{};
        field<PolynomialTerm> terms{};

        Data() = default;
        Data( intfield indices, field<PolynomialBasis> bases, field<unsigned int> site_p, field<PolynomialTerm> terms )
                : indices( std::move( indices ) ),
                  bases( std::move( bases ) ),
                  site_p( std::move( site_p ) ),
                  terms( std::move( terms ) ) {};
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

    struct Index
    {
        int ispin, iani;
    };

    using IndexContainer = Engine::IndexContainer<Biaxial_Anisotropy>;

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

    static void
    applyGeometry( const ::Data::Geometry & geometry, const Data & data, Cache &, IndexContainer & container )
    {
        using Indexing::check_atom_type;
        auto indices = std::vector( geometry.nos, field<Index>{} );
        const auto N = geometry.n_cell_atoms;

        for( int icell = 0; icell < geometry.n_cells_total; ++icell )
        {
            for( int iani = 0; iani < data.indices.size(); ++iani )
            {
                int ispin = icell * N + data.indices[iani];
                if( check_atom_type( geometry.atom_types[ispin] ) )
                    indices[ispin].push_back( Index{ ispin, iani } );
            }
        }

        container = make_index_container<Biaxial_Anisotropy>( std::move( indices ) );
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
              terms( data.terms.data() ) {};

    const bool is_contributing;

protected:
    const PolynomialBasis * bases;
    const unsigned int * site_p;
    const PolynomialTerm * terms;
};

template<>
inline scalar Biaxial_Anisotropy::Energy::operator()( Span<const Index> index, quantity<const Vector3 *> state ) const
{
    using Utility::fastpow;
    if( !is_contributing )
        return 0;
    else
        return Backend::transform_reduce(
            index.begin(), index.end(), scalar( 0 ), Backend::plus<scalar>{},
            [this, state] SPIRIT_LAMBDA( const Index & idx ) -> scalar
            {
                scalar result              = 0;
                const auto & [ispin, iani] = idx;
                const scalar s1            = bases[iani].k1.dot( state.spin[ispin] );
                const scalar s2            = bases[iani].k2.dot( state.spin[ispin] );
                const scalar s3            = bases[iani].k3.dot( state.spin[ispin] );

                const scalar sin_theta_2 = 1 - s1 * s1;

                for( auto iterm = site_p[iani]; iterm < site_p[iani + 1]; ++iterm )
                {
                    const auto & [coeff, n1, n2, n3] = terms[iterm];
                    result += coeff * fastpow( sin_theta_2, n1 ) * fastpow( s2, n2 ) * fastpow( s3, n3 );
                }

                return result;
            } );
}

template<>
inline Vector3
Biaxial_Anisotropy::Gradient::operator()( Span<const Index> index, quantity<const Vector3 *> state ) const
{
    using Utility::fastpow;
    if( !is_contributing )
        return Vector3::Zero();
    else
        return Backend::transform_reduce(
            index.begin(), index.end(), Vector3( Vector3::Zero() ), Backend::plus<Vector3>{},
            [this, state] SPIRIT_LAMBDA( const Index & idx ) -> Vector3
            {
                Vector3 result = Vector3::Zero();

                const auto & [ispin, iani] = idx;
                const auto & [k1, k2, k3]  = bases[iani];

                const scalar s1 = k1.dot( state.spin[ispin] );
                const scalar s2 = k2.dot( state.spin[ispin] );
                const scalar s3 = k3.dot( state.spin[ispin] );

                const scalar sin_theta_2 = 1 - s1 * s1;

                for( auto iterm = site_p[iani]; iterm < site_p[iani + 1]; ++iterm )
                {
                    const auto & [coeff, n1, n2, n3] = terms[iterm];

                    const scalar a = fastpow( s2, n2 );
                    const scalar b = fastpow( s3, n3 );
                    const scalar c = fastpow( sin_theta_2, n1 );

                    result += k1 * ( n1 > 0 ? coeff * a * b * n1 * ( -2.0 * s1 * fastpow( sin_theta_2, n1 - 1 ) ) : 0 );
                    result += k2 * ( n2 > 0 ? coeff * b * c * n2 * fastpow( s2, n2 - 1 ) : 0 );
                    result += k3 * ( n3 > 0 ? coeff * a * c * n3 * fastpow( s3, n3 - 1 ) : 0 );
                }
                return result;
            } );
}

template<>
template<typename Callable>
void Biaxial_Anisotropy::Hessian::operator()(
    Span<const Index> index, const StateType & state, Callable & hessian ) const
{
    using Utility::fastpow;
    if( !is_contributing )
        return;

    Backend::cpu::for_each(
        index.begin(), index.end(),
        [this, state, &hessian]( const Index & idx )
        {
            const auto & [ispin, iani] = idx;
            const auto & [k1, k2, k3]  = bases[iani];

            const scalar s1 = k1.dot( state.spin[ispin] );
            const scalar s2 = k2.dot( state.spin[ispin] );
            const scalar s3 = k3.dot( state.spin[ispin] );

            const scalar st2 = 1 - s1 * s1;

            static constexpr auto safepow = []( const scalar base, int exp )
            { return ( exp <= 0 ) ? 1.0 : fastpow( base, static_cast<unsigned int>( exp ) ); };

            for( auto iterm = site_p[iani]; iterm < site_p[iani + 1]; ++iterm )
            {
                const auto & [coeff, n1, n2, n3] = terms[iterm];

                const scalar a = fastpow( s2, n2 );
                const scalar b = fastpow( s3, n3 );
                const scalar c = fastpow( st2, n1 );

                const scalar p_11 = a * b
                                    * ( -2.0 * n1 * safepow( st2, n1 - 1 )
                                        + 4.0 * n1 * ( n1 - 1 ) * s1 * s1 * safepow( st2, n1 - 2 ) );
                const scalar p_22 = n2 * ( n2 - 1 ) * ( b * c * safepow( s2, n2 - 2 ) );
                const scalar p_33 = n3 * ( n3 - 1 ) * ( a * c * safepow( s3, n3 - 2 ) );
                const scalar p_12 = b * n2 * safepow( s2, n2 - 1 ) * ( -2.0 * n1 * s1 ) * safepow( st2, n1 - 1 );
                const scalar p_13 = a * n3 * safepow( s3, n3 - 1 ) * ( -2.0 * n1 * s1 ) * safepow( st2, n1 - 1 );
                const scalar p_23 = c * n2 * safepow( s2, n2 - 1 ) * n3 * safepow( s3, n3 - 1 );

                for( int alpha = 0; alpha < 3; ++alpha )
                {
                    for( int beta = 0; beta < 3; ++beta )
                    {
                        hessian(
                            3 * ispin + alpha, 3 * ispin + beta,
                            coeff
                                * ( p_11 * k1[alpha] * k1[beta] + p_22 * k2[alpha] * k2[beta]
                                    + p_33 * k3[alpha] * k3[beta]
                                    + p_12 * ( k1[alpha] * k2[beta] + k1[beta] * k2[alpha] )
                                    + p_13 * ( k1[alpha] * k3[beta] + k1[beta] * k3[alpha] )
                                    + p_23 * ( k2[alpha] * k3[beta] + k2[beta] * k3[alpha] ) ) );
                    }
                }
            }
        } );
}

} // namespace Interaction

} // namespace Spin

} // namespace Engine
#endif
