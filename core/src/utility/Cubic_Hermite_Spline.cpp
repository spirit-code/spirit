#include <utility/Cubic_Hermite_Spline.hpp>

#include <cmath>

namespace Utility
{
namespace Cubic_Hermite_Spline
{

// see http://de.wikipedia.org/wiki/Kubisch_Hermitescher_Spline
std::vector<std::vector<scalar>> Interpolate(
    const std::vector<scalar> & x, const std::vector<scalar> & p, const std::vector<scalar> & m, int n_interpolations )
{
    scalar x0, x1, p0, p1, m0, m1, t, h00, h10, h01, h11;
    int idx;

    int n_points = p.size() + ( p.size() - 1 ) * n_interpolations;
    std::vector<std::vector<scalar>> result( 2, std::vector<scalar>( n_points ) );

    for( unsigned int i = 0; i < p.size() - 1; ++i )
    {
        x0 = x[i];
        x1 = x[i + 1];
        p0 = p[i];
        p1 = p[i + 1];
        m0 = m[i];
        m1 = m[i + 1];

        for( int j = 0; j < n_interpolations + 1; ++j )
        {
            t   = j / (scalar)( n_interpolations + 1 );
            h00 = 2 * std::pow( t, 3 ) - 3 * std::pow( t, 2 ) + 1;
            h10 = -2 * std::pow( t, 3 ) + 3 * std::pow( t, 2 );
            h01 = std::pow( t, 3 ) - 2 * std::pow( t, 2 ) + t;
            h11 = std::pow( t, 3 ) - std::pow( t, 2 );

            idx            = i * ( n_interpolations + 1 ) + j;
            result[0][idx] = x0 + t * ( x1 - x0 );
            result[1][idx] = h00 * p0 + h10 * p1 + h01 * m0 * ( x0 - x1 ) + h11 * m1 * ( x0 - x1 );
        }
    }
    result[0].back() = x.back();
    result[1].back() = p.back();

    return result;
}

} // namespace Cubic_Hermite_Spline
} // namespace Utility