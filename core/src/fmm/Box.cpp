#include <fmm/Box.hpp>
#include <fmm/Spherical_Harmonics.hpp>
#include <fmm/Utility.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace SimpleFMM
{

using Utility::get_spherical;
using Utility::idx_from_tupel;
using Utility::multipole_idx;
using Utility::multipole_idx_p;
using Utility::n_moments;
using Utility::n_moments_p;
using Utility::tupel_from_idx;

Box::Box( const vectorfield & pos, int level, int l_max, int degree_local )
        : level( level ), pos( pos ), n_spins( pos.size() ), l_max( l_max )
{
    this->pos_indices.resize( n_spins );
    for( int i = 0; i < n_spins; ++i )
    {
        this->pos_indices[i] = i;
    }
    Get_Boundaries();
    Update( l_max, degree_local );
}

Box::Box( const vectorfield & pos, intfield indices, int level, int l_max, int degree_local )
        : level( level ), pos( pos ), pos_indices( indices ), l_max( l_max ), n_spins( indices.size() )
{
    Get_Boundaries();
    Update( l_max, degree_local );
}

Box::Box( const vectorfield & pos, intfield indices, int level, Vector3 min, Vector3 max, int l_max, int degree_local )
        : level( level ),
          pos( pos ),
          pos_indices( indices ),
          l_max( l_max ),
          n_spins( indices.size() ),
          min( min ),
          max( max )
{
    this->center = 0.5 * ( min + max );
    Update( l_max, degree_local );
}

void Box::Update( int l_max, int degree_local )
{
    this->l_max        = l_max;
    this->degree_local = degree_local;
    this->multipole_moments.resize( n_moments( l_max, l_min ) );
    this->multipole_moments.shrink_to_fit();
    this->local_moments.resize( n_moments_p( degree_local ) );
    this->local_moments.shrink_to_fit();
}

// Divides a Box evenly by cutting it in two in every dimension
// TODO: Use normal1 and normal2 ...
std::vector<Box> Box::Divide_Evenly( int n_dim, Vector3 normal1, Vector3 normal2 )
{
    // std::vector<Box> result(8);
    int n_children = std::pow( 2, n_dim );

    if( this->pos_indices.size() < n_children )
    {
        throw std::invalid_argument(
            "This box can't be further divided! At least one of the children would be empty." );
    }

    std::vector<intfield> result_indices( n_children );
    std::vector<Vector3> min( n_children );
    std::vector<Vector3> max( n_children );
    this->n_children = std::pow( 2, n_dim );

    // Build the indice lists of the new boxes
    intfield tupel( n_dim );
    intfield maxVal( n_dim, 2 );
    for( int idx = 0; idx < pos_indices.size(); idx++ )
    {
        auto & i = pos_indices[idx];
        for( int dim = 0; dim < n_dim; dim++ )
        {
            tupel[dim] = ( pos[i][dim] < center[dim] ) ? 0 : 1;
        }
        result_indices[idx_from_tupel( tupel, maxVal )].push_back( i );
    }

    for( int i = 0; i < this->n_children; i++ )
    {
        tupel_from_idx( i, tupel, maxVal );
        for( int j = 0; j < n_dim; j++ )
        {
            min[i][j] = ( tupel[j] == 0 ) ? this->min[j] : this->center[j];
            max[i][j] = ( tupel[j] == 0 ) ? this->center[j] : this->max[j];
        }
        for( int dir = n_dim; dir <= 2; dir++ )
        {
            min[i][dir] = this->min[dir];
            max[i][dir] = this->max[dir];
        }
    }

    std::vector<Box> result;
    for( int i = 0; i < n_children; i++ )
    {
        result.emplace_back( pos, result_indices[i], level + 1, min[i], max[i], this->l_max, this->degree_local );
    }
    return result;
}

// Computes the extents of a box
void Box::Get_Boundaries()
{
    min = pos[pos_indices[0]];
    max = pos[pos_indices[0]];
    for( int idx = 0; idx < pos_indices.size(); idx++ )
    {
        auto & i = pos_indices[idx];
        for( int dir = 0; dir < 3; dir++ )
        {
            if( min[dir] > pos[i][dir] )
                min[dir] = pos[i][dir];
            if( max[dir] < pos[i][dir] )
                max[dir] = pos[i][dir];
        }
    }
    center = 0.5 * ( max + min );
}

// Check if other_box is a near neighbour of this box
bool Box::Is_Near_Neighbour( Box & other_box )
{
    scalar distance       = ( this->center - other_box.center ).norm();
    scalar diagonal_width = ( this->min - this->max ).norm();
    return distance < 1.1 * diagonal_width;
}

// Test if the MAC is fulfilled between two boxes
bool Box::Fulfills_MAC( Box & other_box )
{
    bool on_the_same_level = ( this->level == other_box.level );
    return on_the_same_level && !Is_Near_Neighbour( other_box );
}

void Box::Evaluate_Near_Field( const vectorfield & spins, const scalarfield & mu_s, vectorfield & gradient )
{
    // TODO check if this is working correctly
    for( int i = 0; i < pos_indices.size(); i++ )
    {
        for( int j = i + 1; j < pos_indices.size(); j++ )
        {
            auto & idx1 = pos_indices[i];
            auto & idx2 = pos_indices[j];
            auto r12    = pos[idx1] - pos[idx2];
            auto r      = r12.norm();
            gradient[idx1] += ( 3 * spins[idx2].dot( r12 ) * r12 / std::pow( r, 5 ) - spins[idx2] / std::pow( r, 3 ) )
                              * mu_s[idx2];
            gradient[idx2] += ( 3 * spins[idx1].dot( r12 ) * r12 / std::pow( r, 5 ) - spins[idx1] / std::pow( r, 3 ) )
                              * mu_s[idx1];
        }
    }
}

Vector3 Box::Evaluate_Directly_At( Vector3 r, vectorfield & spins )
{
    Vector3 result = { 0, 0, 0 };
    for( auto p_idx : pos_indices )
    {
        auto r12 = pos[p_idx] - r;
        auto r   = r12.norm();
        result += 3 * spins[p_idx].dot( r12 ) * r12 / std::pow( r, 5 ) - spins[p_idx] / std::pow( r, 3 );
    }
    return result;
}

Vector3 Box::Evaluate_Far_Field_At( Vector3 r )
{
    Vector3 p_sph;
    get_spherical( r - this->center, p_sph );
    Vector3 temp = { 0, 0, 0 };
    for( int l = 0; l <= l_max; l++ )
    {
        for( int m = -l; m <= l; m++ )
        {
            auto & moment = this->local_moments[multipole_idx_p( l, m )];
            temp += ( moment * Spherical_Harmonics::R( l, m, p_sph[0], p_sph[1], p_sph[2] ) ).real();
        }
    }
    return temp;
}

Vector3c Box::Evaluate_Multipole_Expansion_At( Vector3 r )
{
    Vector3c result = { 0, 0, 0 };
    Vector3 r_sph;
    get_spherical( r - this->center, r_sph );
    for( auto l = l_min; l <= l_max; l++ )
    {
        for( auto m = -l; m <= l; m++ )
        {
            result += this->multipole_moments[multipole_idx( l, m, l_min )]
                      * Spherical_Harmonics::S( l, m, r_sph[0], r_sph[1], r_sph[2] );
        }
    }
    return result;
}

void Box::Clear_Local_Moments()
{
    for( Vector3c & l : this->local_moments )
    {
        l = { 0, 0, 0 };
    }
}

void Box::Clear_Multipole_Moments()
{
    for( Vector3c & m : this->multipole_moments )
    {
        m = { 0, 0, 0 };
    }
}

void Box::Clear_Moments()
{
    this->Clear_Local_Moments();
    this->Clear_Multipole_Moments();
}

// Mainly for debugging
void Box::Print_Info( bool print_multipole_moments, bool print_local_moments )
{
    std::cout << "-------------- Box Info --------------" << std::endl
              << " ID          = " << id << std::endl
              << " Level       = " << level << std::endl
              << " n_children  = " << this->n_children << std::endl
              << " n_particles = " << pos_indices.size() << std::endl
              << " center      = " << center[0] << " " << center[1] << " " << center[2] << " " << std::endl
              << " Min / Max " << std::endl
              << "   x: " << min[0] << " / " << max[0] << std::endl
              << "   y: " << min[1] << " / " << max[1] << std::endl
              << "   z: " << min[2] << " / " << max[2] << std::endl
              << " n_interaction = " << interaction_list.size() << std::endl;
    if( print_multipole_moments )
    {
        std::cout << "== Multipole Moments == " << std::endl;
        for( auto l = l_min; l <= l_max; l++ )
        {
            for( auto m = -l; m <= l; m++ )
            {
                std::cout << ">> --- l = " << l << ", m = " << m << " -- <<" << std::endl;
                std::cout << multipole_moments[multipole_idx( l, m, l_min )] << std::endl;
            }
        }
    }
    if( print_local_moments )
    {
        std::cout << "== Local Moments == " << std::endl;
        for( auto l = 0; l <= l_max; l++ )
        {
            for( auto m = 0; m <= l; m++ )
            {
                std::cout << ">> --- l = " << l << ", m = " << m << " -- <<" << std::endl;
                std::cout << local_moments[multipole_idx_p( l, m )] << std::endl;
            }
        }
    }
    std::cout << "Interaction List: " << std::endl;
    for( auto i : interaction_list )
        std::cout << i << " ";
    std::cout << std::endl;
}

} // namespace SimpleFMM