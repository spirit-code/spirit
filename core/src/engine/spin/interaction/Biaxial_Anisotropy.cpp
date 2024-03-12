#include <engine/spin/interaction/Biaxial_Anisotropy.hpp>
#include <utility/Constants.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

namespace Engine
{

namespace Spin
{

namespace Interaction
{

template<>
scalar Biaxial_Anisotropy::Energy::operator()( const Index & index, const vectorfield & spins ) const
{
    using std::pow;
    scalar result = 0;
    if( !index.has_value() )
        return result;

    const auto & [ispin, iani] = *index;
    const scalar s1            = data.bases[iani].k1.dot( spins[ispin] );
    const scalar s2            = data.bases[iani].k2.dot( spins[ispin] );
    const scalar s3            = data.bases[iani].k3.dot( spins[ispin] );

    const scalar sin_theta_2 = 1 - s1 * s1;

    for( auto iterm = data.site_p[iani]; iterm < data.site_p[iani + 1]; ++iterm )
    {
        const auto & [coeff, n1, n2, n3] = data.terms[iterm];
        result += coeff * pow( sin_theta_2, n1 ) * pow( s2, n2 ) * pow( s3, n3 );
    }

    return result;
}

template<>
Vector3 Biaxial_Anisotropy::Gradient::operator()( const Index & index, const vectorfield & spins ) const
{
    using std::pow;
    Vector3 result = Vector3::Zero();
    if( !index.has_value() )
        return result;

    const auto & [ispin, iani] = *index;
    const auto & [k1, k2, k3]  = data.bases[iani];

    const scalar s1 = k1.dot( spins[ispin] );
    const scalar s2 = k2.dot( spins[ispin] );
    const scalar s3 = k3.dot( spins[ispin] );

    const scalar sin_theta_2 = 1 - s1 * s1;

    for( auto iterm = data.site_p[iani]; iterm < data.site_p[iani + 1]; ++iterm )
    {
        const auto & [coeff, n1, n2, n3] = data.terms[iterm];

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
};

} // namespace Interaction

} // namespace Spin

} // namespace Engine
