#include <utility/Cubic_Hermite_Spline.hpp>

#include <cmath>

namespace Utility
{
namespace Cubic_Hermite_Spline
{

// See http://de.wikipedia.org/wiki/Kubisch_Hermitescher_Spline
std::vector<std::vector<scalar>> Interpolate(
    const std::vector<scalar> & x, const std::vector<scalar> & p, const std::vector<scalar> & m, int n_interpolations )
{
    scalar x0, x1, p0, p1, m0, m1, t, h00, h10, h01, h11;
    std::size_t idx;

    int n_points = p.size() + ( p.size() - 1 ) * n_interpolations;
    std::vector<std::vector<scalar>> result( 2, std::vector<scalar>( n_points ) );

    for( std::size_t i = 0; i < p.size() - 1; ++i )
    {
        x0 = x[i];
        x1 = x[i + 1];
        p0 = p[i];
        p1 = p[i + 1];
        m0 = m[i];
        m1 = m[i + 1];

        for( int j = 0; j < n_interpolations + 1; ++j )
        {
            t   = j / static_cast<scalar>( n_interpolations + 1 );
            h00 = 2 * /*pow(t, 3)*/ t * t * t - 3 * /*pow(t, 2)*/ t * t + 1;
            h10 = -2 * /*pow(t, 3)*/ t * t * t + 3 * /*pow(t, 2)*/ t * t;
            h01 = /*pow(t, 3)*/ t * t * t - 2 * /*pow(t, 2)*/ t * t + t;
            h11 = /*pow(t, 3)*/ t * t * t - /*pow(t, 2)*/ t * t;

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
