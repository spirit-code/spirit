#pragma once
#ifndef SIMPLE_FMM_Tree_HPP
#define SIMPLE_FMM_Tree_HPP

#include <fmm/Box.hpp>
#include <fmm/SimpleFMM_Defines.hpp>

#include <iterator>
#include <vector>

namespace SimpleFMM
{

struct Pair_of_Boxes
{
    Pair_of_Boxes( Box & box1, Box & box2 ) : box1( box1 ), box2( box2 )
    {
        if( &box1 == &box2 )
        {
            for( int i = 0; i < box1.pos_indices.size(); i++ )
            {
                for( int j = i; j < box2.pos_indices.size(); j++ )
                {
                    auto & diff = box1.pos[box1.pos_indices[i]] - box2.pos[box2.pos_indices[j]];
                    if( diff.norm() > 1e-10 )
                    {
                        directions.push_back( diff / diff.norm() );
                        magnitudes3.push_back( diff.norm() * diff.norm() * diff.norm() );
                    }
                    else
                    {
                        directions.push_back( { 0, 0, 0 } );
                        magnitudes3.push_back( { 1 } );
                    }
                }
            }
        }
        else
        {
            for( int i = 0; i < box1.pos_indices.size(); i++ )
            {
                for( int j = 0; j < box2.pos_indices.size(); j++ )
                {
                    auto & diff = box1.pos[box1.pos_indices[i]] - box2.pos[box2.pos_indices[j]];
                    if( diff.norm() > 1e-10 )
                    {
                        directions.push_back( diff / diff.norm() );
                        magnitudes3.push_back( diff.norm() * diff.norm() * diff.norm() );
                    }
                    else
                    {
                        directions.push_back( { 0, 0, 0 } );
                        magnitudes3.push_back( { 1 } );
                    }
                }
            }
        }
    }

    void Interact_Directly(
        const vectorfield & spins, const scalarfield & mu_s, vectorfield & gradient, scalar prefactor = 1 )
    {
        int count = 0;
        if( &box1 == &box2 )
        {
            for( int i = 0; i < box1.pos_indices.size(); i++ )
            {
                for( int j = i; j < box2.pos_indices.size(); j++ )
                {
                    auto & idx1 = box1.pos_indices[i];
                    auto & idx2 = box2.pos_indices[j];
                    if( idx1 != idx2 )
                    {
                        gradient[idx1]
                            += prefactor * mu_s[idx2] * mu_s[idx1]
                               * ( 3 * spins[idx2].dot( directions[count] ) * directions[count] - spins[idx2] )
                               / magnitudes3[count];
                        gradient[idx2]
                            += prefactor * mu_s[idx1] * mu_s[idx2]
                               * ( 3 * spins[idx1].dot( directions[count] ) * directions[count] - spins[idx1] )
                               / magnitudes3[count];
                    }
                    count++;
                }
            }
        }
        else
        {
            for( int i = 0; i < box1.pos_indices.size(); i++ )
            {
                for( int j = 0; j < box2.pos_indices.size(); j++ )
                {
                    auto & idx1 = box1.pos_indices[i];
                    auto & idx2 = box2.pos_indices[j];
                    if( idx1 != idx2 )
                    {
                        gradient[idx1]
                            += prefactor * mu_s[idx2] * mu_s[idx2]
                               * ( 3 * spins[idx2].dot( directions[count] ) * directions[count] - spins[idx2] )
                               / magnitudes3[count];
                        gradient[idx2]
                            += prefactor * mu_s[idx1] * mu_s[idx1]
                               * ( 3 * spins[idx1].dot( directions[count] ) * directions[count] - spins[idx1] )
                               / magnitudes3[count];
                    }
                    count++;
                }
            }
        }
    }

    Box & box1;
    Box & box2;
    vectorfield directions;
    scalarfield magnitudes3;
};

class Tree
{
    // TODO
    // 1. add functions that return **iterators** over
    //  levels
    //  children
    //  near_neighbours

    using iterator = Box *;

    std::vector<Box> boxes;
    intfield start_idx_level;
    intfield n_boxes_on_level;
    int _Get_Parent_Idx( int idx );
    int _Get_Child_Idx( int idx );

public:
    int div;
    int n_level;
    int children_per_box;
    int n_boxes;
    int n_dim;
    int l_min;
    int l_max;
    int degree_local;

    std::vector<Pair_of_Boxes> direct_interaction_pairs;

    Tree();
    Tree( int depth, vectorfield & pos, int n_dim = 3, int l_max = 6, int degree_local = 3 );

    // Implement all these
    iterator begin_level( int level )
    {
        return iterator( &boxes[start_idx_level[level]] );
    };

    iterator end_level( int level )
    {
        return iterator( &boxes[start_idx_level[level] + n_boxes_on_level[level]] );
    };

    iterator begin_children( Box & box )
    {
        return iterator( &boxes[_Get_Child_Idx( box.id )] );
    };

    iterator end_children( Box & box )
    {
        return iterator( &boxes[_Get_Child_Idx( box.id ) + box.n_children] );
    };

    // TODO
    iterator begin_near_neighbours( Box & box );
    iterator end_near_neighbours( Box & box );

    Box & Get_Box( int idx );
    Box & Get_Parent( Box & box );
    void Build_Caches( Box & box );

    void Upward_Pass( const vectorfield & spins, const scalarfield & mu_s );
    void Cleaning_Pass();
    void Downward_Pass();
    void
    Evaluation( const vectorfield & spins, const scalarfield & mu_s, vectorfield & gradient, scalar prefactor = 1 );
    void Direct_Evaluation( const vectorfield & spins, const scalarfield & mu_s, vectorfield & gradient );
};

} // namespace SimpleFMM

#endif