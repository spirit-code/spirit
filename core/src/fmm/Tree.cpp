#include <fmm/Formulas.hpp>
#include <fmm/Spherical_Harmonics.hpp>
#include <fmm/Tree.hpp>
#include <fmm/Utility.hpp>

#include <cmath>
#include <iostream>

namespace SimpleFMM
{

using Utility::minus_one_power;

Tree::Tree(){};

// The Tree constructor does the following
//  1. Create a tree with n_level levels of boxes
//  2. Each box on level l gets divided into 8 equally sized boxes from which level l+1 is formed
//  3. Determine the interaction lists according to the MAC
//  4. The boxes on the deepest level calculate the hessian of the multipole moments

Tree::Tree( int n_level, vectorfield & pos, int n_dim, int l_max, int degree_local )
        : n_level( n_level ), l_min( 2 ), l_max( l_max ), degree_local( degree_local ), n_dim( n_dim )
{
    this->n_boxes          = 0;
    this->children_per_box = std::pow( 2, n_dim );

    // Could be replaced by formula for geometric sum
    for( int i = 0; i < n_level; i++ )
    {
        n_boxes += std::pow( children_per_box, i );
    }

    // This is important to avoid iterator invalidation on boxes.push_back(..)
    boxes.reserve( n_boxes );

    // Push back the root box
    this->boxes.push_back( Box( pos, 0, l_max, degree_local ) );
    start_idx_level.push_back( 0 );
    n_boxes_on_level.push_back( 1 );
    Get_Box( 0 ).id = 0;
    Get_Box( 0 ).Get_Boundaries();

    for( int level = 1; level < n_level; level++ )
    {
        start_idx_level.push_back( boxes.size() );
        // Push back the children of all the boxes on the previous level
        for( auto it = begin_level( level - 1 ); it != end_level( level - 1 ); it++ )
        {
            for( auto box : it->Divide_Evenly( n_dim ) )
            {
                box.id = boxes.size();
                boxes.push_back( box );
            }
        }
        n_boxes_on_level.push_back( boxes.size() - start_idx_level[level] );

        // Build the interaction lists
        // TODO: more efficient implementation
        // Iterate over the parent level
        for( auto it_par = begin_level( level - 1 ); it_par != end_level( level - 1 ); it_par++ )
        {
            // Find boxes at the parent level which are near neighbours
            for( auto it_par_2 = begin_level( level - 1 ); it_par_2 != end_level( level - 1 ); it_par_2++ )
            {
                if( it_par->Is_Near_Neighbour( *it_par_2 ) )
                {
                    // If the children fulfill the mac at this level add them to the interactions list
                    for( auto it_ch = begin_children( *it_par ); it_ch != end_children( *it_par ); it_ch++ )
                    {
                        for( auto it_ch_2 = begin_children( *it_par_2 ); it_ch_2 != end_children( *it_par_2 );
                             it_ch_2++ )
                        {
                            if( it_ch->Fulfills_MAC( *it_ch_2 ) )
                            {
                                it_ch->interaction_list.push_back( it_ch_2->id );
                            }
                        }
                    }
                }
            }
        }
    }

    for( auto box = begin_level( 0 ); box != end_level( n_level - 1 ); box++ )
    {
        this->Build_Caches( *box );
    }

    // The boxes on the last level calculate the hessian of their estatic multipole moments and find the indices of
    // their near neighbours
    for( auto it_last = begin_level( n_level - 1 ); it_last != end_level( n_level - 1 ); it_last++ )
    {
        Get_Multipole_Hessians( *it_last, l_min, l_max, 1e-3 );
        Build_Far_Field_Cache( *it_last, degree_local );
        for( auto it_last2 = it_last; it_last2 != end_level( n_level - 1 ); it_last2++ )
        {
            if( it_last->Is_Near_Neighbour( *it_last2 ) )
            {
                this->direct_interaction_pairs.emplace_back( Pair_of_Boxes( *it_last, *it_last2 ) );
            }
        }
    }
}

void Tree::Upward_Pass( const vectorfield & spins, const scalarfield & mu_s )
{
    this->Cleaning_Pass();
    for( auto box = this->begin_level( n_level - 1 ); box != this->end_level( n_level - 1 ); box++ )
    {
        Calculate_Multipole_Moments( *box, spins, mu_s, l_min, l_max );
    }
    for( int lvl = n_level - 2; lvl >= 0; lvl-- )
    {
        for( auto box = this->begin_level( lvl ); box != this->end_level( lvl ); box++ )
        {
            for( auto c = begin_children( *box ); c != end_children( *box ); c++ )
            {
                Add_Multipole_Moments( *box, *c, l_min, l_max );
            }
        }
    }
}

// Performs a naive O(N^2) summation for the gradient
void Tree::Direct_Evaluation( const vectorfield & spins, const scalarfield & mu_s, vectorfield & gradient )
{
    boxes[0].Evaluate_Near_Field( spins, mu_s, gradient );
}

void Tree::Cleaning_Pass()
{
    for( int lvl = 0; lvl < n_level; lvl++ )
    {
        for( auto box = begin_level( lvl ); box != end_level( lvl ); box++ )
        {
            box->Clear_Moments();
        }
    }
}

void Tree::Downward_Pass()
{
    // From the coarset level to the finest
    for( int lvl = 0; lvl < n_level; lvl++ )
    {
        // Each box calculates its local expansion due to the multipole expansions of the boxes in its interaction list
        for( auto box = begin_level( lvl ); box != end_level( lvl ); box++ )
        {
            for( auto interaction_id : box->interaction_list )
            {
                M2L( *box, boxes[interaction_id], l_min, l_max, degree_local );
            }

            // Then the local expansions get translated down to the children
            if( lvl != n_level - 1 )
                for( auto child = begin_children( *box ); child != end_children( *box ); child++ )
                    // box->Add_Local_Moments(*child);
                    Add_Local_Moments( *box, *child, degree_local );
        }
    }
}

void Tree::Evaluation( const vectorfield & spins, const scalarfield & mu_s, vectorfield & gradient, scalar prefactor )
{
    for( auto leaf_box = begin_level( n_level - 1 ); leaf_box != end_level( n_level - 1 ); leaf_box++ )
    {
        Evaluate_Far_Field( *leaf_box, gradient, mu_s, degree_local, prefactor );
    }
    for( auto pair : this->direct_interaction_pairs )
    {
        pair.Interact_Directly( spins, mu_s, gradient, prefactor );
    }
}

// Helper for M2L
std::complex<scalar> _M2L_prefactor1( Vector3 diff_sph, int l, int lp, int m, int mp )
{
    return minus_one_power( lp ) * Spherical_Harmonics::S( l + lp, m + mp, diff_sph[0], diff_sph[1], diff_sph[2] );
}

// Build Cached function values for M2M, M2L, L2L
void Tree::Build_Caches( Box & box )
{
    // M2M Cache
    if( box.level != n_level - 1 )
    {
        for( auto child = begin_children( box ); child != end_children( box ); child++ )
        {
            Cache_M2M_values( box, *child, l_max );
        }
    }

    // L2L Cache
    if( box.level != n_level - 1 )
    {
        for( auto child = begin_children( box ); child != end_children( box ); child++ )
        {
            Cache_L2L_values( box, *child, l_max );
        }
    }

    // M2L Cache
    for( auto interaction_idx : box.interaction_list )
    {
        Box & source_box = this->boxes[interaction_idx];
        Cache_M2L_values( box, source_box, l_max, degree_local, l_min );
    }
}

int Tree::_Get_Parent_Idx( int idx )
{
    int dist_to_start = idx - start_idx_level[boxes[idx].level];
    return dist_to_start / children_per_box + start_idx_level[boxes[idx].level - 1];
}

int Tree::_Get_Child_Idx( int idx )
{
    int dist_to_start = idx - start_idx_level[boxes[idx].level];
    return start_idx_level[boxes[idx].level + 1] + children_per_box * dist_to_start;
}

Box & Tree::Get_Box( int idx )
{
    return this->boxes[idx];
}

Box & Tree::Get_Parent( Box & box )
{
    return this->boxes[_Get_Parent_Idx( box.id )];
}

} // namespace SimpleFMM