#pragma once
#ifndef SimpleFMM_TESTING_UTIL_HPP
#define SimpleFMM_TESTING_UTIL_HPP

#include <fmm/SimpleFMM_Defines.hpp>

#include <fmt/format.h>

namespace Testing
{

using SimpleFMM::intfield;
using SimpleFMM::scalarfield;
using SimpleFMM::Vector3;
using SimpleFMM::vectorfield;

// Lattice Descriptors
enum Lattice
{
    SC
};

// Generates a geometry in a vectorfield pos, of Nabc[0] basis cells in direction 'a' etc.
//     'offset' is the origin of the 0,0,0 basis cell
//     'lattice' speficies the lattice type
//     'lattice_constant' is the lattice constant ..
void generate_geometry(
    vectorfield & pos, intfield Nabc, Vector3 offset = { 0, 0, 0 }, Lattice lattice = Lattice::SC,
    double lattice_constant = 1.0 )
{
    auto & Na = Nabc[0];
    auto & Nb = Nabc[1];
    auto & Nc = Nabc[2];

    if( pos.size() != Na * Nb * Nc )
        pos.resize( Na * Nb * Nc );

    Vector3 ta, tb, tc;
    if( lattice == Lattice::SC )
    {
        ta = { 1, 0, 0 };
        tb = { 0, 1, 0 };
        tc = { 0, 0, 1 };
    }

    for( int c = 0; c < Nc; c++ )
    {
        for( int b = 0; b < Nb; b++ )
        {
            for( int a = 0; a < Na; a++ )
            {
                pos[a + Na * ( b + Nb * c )] = lattice_constant * ( ( a * ta + b * tb + c * tc ) + offset );
            }
        }
    }
}

// Generates 'n_spins' spins and mu_s
//     If a dir != 0 is set the spins point in that direction otherwise they are randomised
//     If 'normalized' is true the spins are normalized
//     mu_s is set to mu_s_value for every spin
void generate_spins(
    vectorfield & spins, scalarfield & mu_s, int n_spins, double mu_s_value = 1.0, Vector3 dir = { 0, 0, 0 },
    bool normalize = true )
{
    if( spins.size() != n_spins )
        spins.resize( n_spins );
    if( mu_s.size() != n_spins )
        mu_s.resize( n_spins );

    bool random = false;
    if( dir.norm() < 1e-8 )
        random = true;

    for( int idx = 0; idx < n_spins; idx++ )
    {
        Vector3 new_spin;
        if( random )
            new_spin = { std::rand(), std::rand(), std::rand() };
        else
            new_spin = dir;

        if( normalize || random )
            new_spin.normalize();
        spins[idx] = new_spin;
        mu_s[idx]  = mu_s_value;
    }
}

// void get_indices( intfield & out, int from, int to )
// {
//     out.resize( to - from + 1 );
//     for( int i = from; i <= to; i++ )
//     {
//         out[i] = i;
//     }
// }

// void print_field( vectorfield field, int from = 0, int to = -1 )
// {
//     int end = ( to >= 0 ) ? to : field.size() - 1;

//     std::cout << "     -------------" << std::endl;
//     for( int i = from; i <= end; i++ )
//     {
//         fmt::print("  [{}]:   {}, {}, {}\n", i, field[i][0], field[i][1], field[i][2]);
//     }
//     std::cout << "     -------------" << std::endl;
// }

// void print_field( intfield field, int from = 0, int to = -1 )
// {
//     int end = ( to >= 0 ) ? to : field.size() - 1;
//     std::cout << "     -------------" << std::endl;
//     for( int i = from; i <= end; i++ )
//     {
//         fmt::print("  [{}]:   {}\n", i, field[i]);
//     }
//     std::cout << "     -------------" << std::endl;
// }

// void print_field( scalarfield field, int from = 0, int to = -1 )
// {
//     int end = ( to >= 0 ) ? to : field.size() - 1;
//     std::cout << "     -------------" << std::endl;
//     for( int i = from; i <= end; i++ )
//     {
//         fmt::print("  [{}]:   {}\n", i, field[i]);
//     }
//     std::cout << "     -------------" << std::endl;
// }

void print_vector( Vector3 vec, int precision = 10 )
{
    fmt::print( "({:.{0}}f}, {:.{0}}f}, {:.{0}}f})\n", precision, vec[0], vec[1], vec[2] );
}

// Caluclates the gradient at 'r' directly
Vector3 calculate_gradient_directly(
    const Vector3 r, const vectorfield & pos, const vectorfield & spins, const scalarfield & mu_s )
{
    Vector3 result = { 0, 0, 0 };
    for( int i = 0; i < spins.size(); i++ )
    {
        Vector3 d = ( pos[i] - r );
        result += mu_s[i] * ( 3 * spins[i].dot( d.normalized() ) * d.normalized() - spins[i] )
                  / ( d.norm() * d.norm() * d.norm() );
    }
    return result;
}

} // namespace Testing

#endif