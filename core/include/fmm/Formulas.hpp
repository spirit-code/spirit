#pragma once
#ifndef SIMPLE_FMM_FORMULAS
#define SIMPLE_FMM_FORMULAS

#include <fmm/Box.hpp>
#include <fmm/Spherical_Harmonics.hpp>

namespace SimpleFMM
{

using Utility::get_spherical;
using Utility::minus_one_power;
using Utility::multipole_idx;
using Utility::multipole_idx_p;
using Utility::n_moments;
using Utility::n_moments_p;

// ============================================
//               P2M functions
// ============================================
inline void Get_Multipole_Hessians( Box & box, int l_min, int l_max, scalar epsilon = 1e-3 )
{
    for( int l = l_min; l <= l_max; l++ )
    {
        for( int m = -l; m <= l; m++ )
        {
            for( auto p_idx : box.pos_indices )
            {
                // Calculate the hessian via finite difference
                Matrix3c hessian;
                vectorfield d_xyz = { { 0.5 * epsilon, 0, 0 }, { 0, 0.5 * epsilon, 0 }, { 0, 0, 0.5 * epsilon } };

                Vector3 spherical;
                // Fill the hessian with second derivatives
                for( int dir1 = 0; dir1 < 3; dir1++ )
                {
                    for( int dir2 = dir1; dir2 < 3; dir2++ )
                    {
                        get_spherical( box.pos[p_idx] + d_xyz[dir1] + d_xyz[dir2] - box.center, spherical );
                        auto fpp
                            = std::conj( Spherical_Harmonics::R( l, m, spherical[0], spherical[1], spherical[2] ) );

                        get_spherical( box.pos[p_idx] + d_xyz[dir1] - d_xyz[dir2] - box.center, spherical );
                        auto fpm
                            = std::conj( Spherical_Harmonics::R( l, m, spherical[0], spherical[1], spherical[2] ) );

                        get_spherical( box.pos[p_idx] - d_xyz[dir1] + d_xyz[dir2] - box.center, spherical );
                        auto fmp
                            = std::conj( Spherical_Harmonics::R( l, m, spherical[0], spherical[1], spherical[2] ) );

                        get_spherical( box.pos[p_idx] - d_xyz[dir1] - d_xyz[dir2] - box.center, spherical );
                        auto fmm
                            = std::conj( Spherical_Harmonics::R( l, m, spherical[0], spherical[1], spherical[2] ) );

                        hessian( dir1, dir2 ) = 1 / ( epsilon * epsilon ) * ( fpp - fpm - fmp + fmm );
                        if( dir1 != dir2 )
                            hessian( dir2, dir1 ) = hessian( dir1, dir2 );
                    }
                }
                box.multipole_hessians.push_back( hessian );
            }
        }
    }
}

inline void
Calculate_Multipole_Moments( Box & box, const vectorfield & spins, const scalarfield & mu_s, int l_min, int l_max )
{
    assert( box.multipole_moments.size() == n_moments( l_max, l_min ) );
    for( int l = l_min; l <= l_max; l++ )
    {
        for( int m = -l; m <= l; m++ )
        {
            for( int i = 0; i < box.pos_indices.size(); i++ )
            {
                auto p_idx = box.pos_indices[i];
                box.multipole_moments[multipole_idx( l, m, l_min )]
                    += box.multipole_hessians[multipole_idx( l, m, l_min ) * box.pos_indices.size() + i] * spins[p_idx]
                       * mu_s[p_idx];
            }
        }
    }
}

// ============================================
//                M2M functions
// ============================================
inline void Cache_M2M_values( Box & parent_box, const Box & child_box, int l_max )
{
    parent_box.M2M_cache[child_box.id] = complexfield( n_moments( l_max ) * parent_box.n_children );
    Vector3 diff_sph;
    get_spherical( child_box.center - parent_box.center, diff_sph );
    for( int l = 0; l <= l_max; l++ )
    {
        for( int m = -l; m <= l; m++ )
        {
            parent_box.M2M_cache[child_box.id][multipole_idx( l, m )]
                = std::conj( Spherical_Harmonics::R( l, m, diff_sph[0], diff_sph[1], diff_sph[2] ) );
        }
    }
}

inline void Add_Multipole_Moments( Box & parent_box, const Box & child_box, int l_min, int l_max )
{
    auto diff = child_box.center - parent_box.center;
    // Vector3 diff_sph;
    // get_spherical(diff, diff_sph);
    for( int l = l_min; l <= l_max; l++ )
    {
        for( int m = -l; m <= l; m++ )
        {
            for( int lc = l_min; lc <= l; lc++ )
            {
                for( int mc = std::max( -lc, m + lc - l ); mc <= std::min( lc, m + l - lc );
                     mc++ ) // because we need |m-mc| <= l-lc
                {
                    parent_box.multipole_moments[multipole_idx( l, m, l_min )]
                        += child_box.multipole_moments[multipole_idx( lc, mc, l_min )]
                           * parent_box.M2M_cache[child_box.id][multipole_idx( l - lc, m - mc )];
                }
            }
        }
    }
}

// ============================================
//               M2L functions
// ============================================
inline void Cache_M2L_values( Box & target_box, const Box & source_box, int l_max, int degree_local, int l_min )
{
    target_box.M2L_cache[source_box.id] = MatrixXc( n_moments_p( degree_local ), n_moments( l_max, l_min ) );
    Vector3 diff_sph;
    get_spherical( target_box.center - source_box.center, diff_sph );
    for( int l = 0; l <= degree_local; l++ )
    {
        for( int m = 0; m <= l; m++ )
        {
            int row = multipole_idx_p( l, m );
            for( int lp = l_min; lp <= l_max; lp++ )
            {
                for( int mp = -lp; mp <= lp; mp++ )
                {
                    int col = multipole_idx( lp, mp, l_min );
                    // target_box.M2L_cache[source_box.id](row, col) = Spherical_Harmonics::O(l+lp, -(m+mp),
                    // diff_sph[0], diff_sph[1], diff_sph[2]);
                    target_box.M2L_cache[source_box.id]( row, col )
                        = minus_one_power( l + m )
                          * Spherical_Harmonics::S( l + lp, mp - m, diff_sph[0], diff_sph[1], diff_sph[2] );
                }
            }
        }
    }
}

// Transform multipole moments of source_Box into local moments around target_box center
inline void M2L( Box & target_box, const Box & source_box, int l_min, int l_max, int degree_local )
{
    // Fetch the transfer matrix
    using MatrixXc_row = Eigen::Matrix<std::complex<scalar>, Eigen::Dynamic, 3, Eigen::RowMajor>;

    MatrixXc & T = target_box.M2L_cache[source_box.id];

    const std::complex<double> * raw_data_ptr_in
        = reinterpret_cast<const std::complex<scalar> *>( source_box.multipole_moments.data() );
    std::complex<double> * raw_data_ptr_out
        = reinterpret_cast<std::complex<scalar> *>( target_box.local_moments.data() );

    auto in_map = Eigen::Map<const MatrixXc_row>( raw_data_ptr_in, n_moments( l_max, l_min ), 3 );

    auto out_map = Eigen::Map<MatrixXc_row>( raw_data_ptr_out, n_moments_p( degree_local ), 3 );

    out_map += T * in_map;
}

// inline void Cache_M2L_values(Box& target_box, const Box& source_box, int l_max)
// {
//     target_box.M2L_cache[source_box.id] = complexfield(n_moments(2*l_max));
//     Vector3 diff_sph;
//     get_spherical(target_box.center - source_box.center, diff_sph);
//     for(int l = 0; l <= 2*l_max; l++)
//     {
//         for(int m = -l; m <= l; m++)
//         {
//             target_box.M2L_cache[source_box.id][multipole_idx(l, m)] = Spherical_Harmonics::S(l, m, diff_sph[0],
//             diff_sph[1], diff_sph[2]);
//         }
//     }
// }

// //Transform multipole moments of source_Box into local moments around target_box center
// inline void M2L(Box& target_box, const Box& source_box, int l_min, int l_max, int degree_local)
// {
//     for(int lp = 0; lp <= degree_local; lp++)
//     {
//         for(int mp = 0; mp <= lp; mp++)
//         {
//             Vector3c temp;
//             for(int l = l_min; l <= l_max; l++)
//             {
//                 for(int m = -l; m < l+1; m++)
//                 {
//                     target_box.local_moments[multipole_idx_p(lp, mp)] += minus_one_power(lp) *
//                     (target_box.M2L_cache[source_box.id][multipole_idx(l+lp, m+mp)] *
//                     source_box.multipole_moments[multipole_idx(l, m, l_min)]).conjugate();
//                 }
//             }
//         }
//     }
// }

// ============================================
//               L2L functions
// ============================================

// Cache the transition functions for
inline void Cache_L2L_values( Box & parent_box, const Box & child_box, int l_max )
{
    parent_box.L2L_cache[child_box.id] = complexfield( n_moments( 2 * l_max ) );
    Vector3 diff_sph;
    get_spherical( child_box.center - parent_box.center, diff_sph );
    for( int l = 0; l <= 2 * l_max; l++ )
    {
        for( int m = -l; m <= l; m++ )
        {
            parent_box.L2L_cache[child_box.id][multipole_idx( l, m )]
                = std::conj( Spherical_Harmonics::R( l, m, diff_sph[0], diff_sph[1], diff_sph[2] ) );
        }
    }
}

// Add local moments of parent_box to child_box
inline void Add_Local_Moments( const Box & parent_box, Box & child_box, int degree_local )
{
    assert( child_box.local_moments.size() == n_moments_p( degree_local ) );
    Vector3 diff_sph;
    get_spherical( child_box.center - parent_box.center, diff_sph );
    for( int l = 0; l <= degree_local; l++ )
    {
        for( int m = 0; m <= l; m++ )
        {
            for( int lp = l; lp <= degree_local; lp++ )
            {
                for( int mp = -lp; mp <= lp; mp++ )
                {
                    if( lp - l >= std::abs( mp - m ) )
                    {
                        auto moment = ( mp < 0 ) ?
                                          minus_one_power( mp )
                                              * parent_box.local_moments[multipole_idx_p( lp, -mp )].conjugate() :
                                          parent_box.local_moments[multipole_idx_p( lp, mp )];
                        child_box.local_moments[multipole_idx_p( l, m )]
                            += moment * Spherical_Harmonics::R( lp - l, mp - m, diff_sph[0], diff_sph[1], diff_sph[2] );
                    }
                }
            }
        }
    }
}

// ============================================
//             Evaluation functions
// ============================================
inline void Build_Far_Field_Cache( Box & box, int degree_local )
{
    for( int i = 0; i < box.n_spins; i++ )
    {
        auto & p_idx = box.pos_indices[i];
        Vector3 p_sph;
        get_spherical( box.pos[p_idx] - box.center, p_sph );
        for( int l = 0; l <= degree_local; l++ )
        {
            for( int m = 0; m <= l; m++ )
            {
                box.Farfield_cache.push_back( Spherical_Harmonics::R( l, m, p_sph[0], p_sph[1], p_sph[2] ) );
            }
        }
    }
}

inline void Evaluate_Far_Field(
    const Box & box, vectorfield & gradient, const scalarfield mu_s, int degree_local, scalar prefactor = 1 )
{
    for( int i = 0; i < box.n_spins; i++ )
    {
        auto & p_idx = box.pos_indices[i];
        for( int l = 0; l <= degree_local; l++ )
        {
            gradient[p_idx] += prefactor * mu_s[p_idx]
                               * ( box.local_moments[multipole_idx_p( l, 0 )]
                                   * box.Farfield_cache[n_moments_p( degree_local ) * i + multipole_idx_p( l, 0 )] )
                                     .real();
            for( int m = 1; m <= l; m++ )
            {
                auto & moment = box.local_moments[multipole_idx_p( l, m )];
                auto & cache  = box.Farfield_cache[n_moments_p( degree_local ) * i + multipole_idx_p( l, m )];
                gradient[p_idx] += prefactor * mu_s[p_idx] * 2 * ( moment * cache ).real();
            }
        }
    }
}

inline Vector3 Evaluate_Multipole_Expansion_At( Vector3 r, const Box & box, int l_min, int l_max )
{
    Vector3 result = { 0, 0, 0 };
    Vector3 r_sph;
    get_spherical( r - box.center, r_sph );
    for( auto l = l_min; l <= l_max; l++ )
    {
        for( auto m = -l; m <= l; m++ )
        {
            result += ( box.multipole_moments[multipole_idx( l, m, l_min )]
                        * Spherical_Harmonics::S( l, m, r_sph[0], r_sph[1], r_sph[2] ) )
                          .real();
        }
    }
    return result;
}

} // namespace SimpleFMM

#endif