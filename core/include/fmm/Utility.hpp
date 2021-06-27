#pragma once
#ifndef SIMPLE_FMM_UTILITY_HPP
#define SIMPLE_FMM_UTILITY_HPP

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace SimpleFMM
{
namespace Utility
{

// Calculate (a!/b!)
inline scalar factorial( int a, int b = 1 )
{
    assert( a >= 0 && b >= 0 );
    scalar result = 1;
    for( int i = std::min( a, b ) + 1; i <= std::max( a, b ); i++ )
        result *= i;
    return ( a > b ) ? result : 1 / result;
}

// Compute (-1)^exponent
inline scalar minus_one_power( int exponent )
{
    // Compiler optimization should take care of the modulo
    return ( exponent % 2 == 0 ) ? 1.0 : -1.0;
}

// Compute (i)^exponent
inline std::complex<scalar> i_power( int exponent )
{
    auto mod4 = exponent % 4;
    if( mod4 == 0 )
        return std::complex<scalar>( 1, 0 );
    else if( mod4 == 1 )
        return std::complex<scalar>( 0, 1 );
    else if( mod4 == 2 )
        return std::complex<scalar>( -1, 0 );
    else // mod4 == 3
        return std::complex<scalar>( 0, -1 );
}

inline void get_cartesian( const Vector3 & spherical_in, Vector3 & cartesian_out )
{
    cartesian_out[0] = spherical_in[0] * std::cos( spherical_in[1] ) * std::sin( spherical_in[2] );
    cartesian_out[1] = spherical_in[0] * std::sin( spherical_in[1] ) * std::sin( spherical_in[2] );
    cartesian_out[2] = spherical_in[0] * std::cos( spherical_in[2] );
}

// Calculates spherical coordinates from cartesian input
//     spherical_out[0] (r)     - radius
//     spherical_out[1] (phi)   - azimuth angle  [0, 2pi]
//     spherical_out[2] (theta) - latitude angle [0, pi]
inline void get_spherical( const Vector3 & cartesian_in, Vector3 & spherical_out )
{
    // r, phi, theta
    spherical_out[0] = cartesian_in.norm();
    if( spherical_out[0] > 1e-10 )
    {
        spherical_out[1] = std::atan2( cartesian_in[1], cartesian_in[0] );
        spherical_out[2] = std::acos( cartesian_in[2] / spherical_out[0] );
    }
    else
    {
        spherical_out[1] = 0;
        spherical_out[2] = 0;
    }
    if( spherical_out[1] < 0 )
        spherical_out[1] += 2 * 3.14159265;
}

// Give the linear index of the multipole moment with l and m
// Counted like  l    m   idx
//               0    0   0
//               1   -1   1
//               1    0   2
//               1    1   3
//               2   -2   4
//               etc
inline int multipole_idx( int l, int m, int l_min = 0 )
{
    assert( l >= std::abs( m ) );
    assert( l >= l_min );
    return m + l * ( l + 1 ) - l_min * l_min;
}

inline int n_moments( int l_max, int l_min = 0 )
{
    assert( l_max >= 0 );
    assert( l_min >= 0 );
    assert( l_max >= l_min );
    return multipole_idx( l_max, l_max, l_min ) + 1;
}

inline int multipole_idx_p( int l, int m, int l_min = 0 )
{
    assert( l >= std::abs( m ) );
    assert( l >= l_min );
    assert( m >= 0 );
    return m + l * ( l + 1 ) / 2 - l_min * ( l_min + 1 ) / 2;
}

inline int n_moments_p( int l_max, int l_min = 0 )
{
    assert( l_max >= 0 );
    assert( l_min >= 0 );
    assert( l_max >= l_min );
    return multipole_idx_p( l_max, l_max, l_min ) + 1;
}

// reverse of multipole_idx
// inline int multipole_from_idx(int idx, int & l, int & m)
// {
//     //TODO
// }
// Get the linear index in a n-D array where tupel contains the components in n-dimensions from fastest to slowest
// varying and maxVal is the extent in every dimension
inline int idx_from_tupel( const std::vector<int> & tupel, const std::vector<int> & maxVal )
{
    int idx  = 0;
    int mult = 1;
    for( int i = 0; i < tupel.size(); i++ )
    {
        idx += mult * tupel[i];
        mult *= maxVal[i];
    }
    return idx;
}

// reverse of idx_from_tupel
inline void tupel_from_idx( int & idx, std::vector<int> & tupel, std::vector<int> & maxVal )
{
    int idx_diff = idx;
    int div      = 1;
    for( int i = 0; i < tupel.size() - 1; i++ )
        div *= maxVal[i];
    for( int i = tupel.size() - 1; i > 0; i-- )
    {
        tupel[i] = idx_diff / div;
        idx_diff -= tupel[i] * div;
        div /= maxVal[i - 1];
    }
    tupel[0] = idx_diff / div;
}

} // namespace Utility
} // namespace SimpleFMM

#endif