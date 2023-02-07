#ifndef SPIRIT_CORE_PAIR_SORTING_HPP
#define SPIRIT_CORE_PAIR_SORTING_HPP
#include "Spirit_Defines.h"
#include <fmt/format.h>
#include <algorithm>
#include <engine/Vectormath_Defines.hpp>
#include <numeric>

namespace Engine
{

struct Pair_Order
{
    field<int> n_pairs_per_cell_atom;
    field<int> offset_per_cell_atom;
    std::vector<int> indices;

    const field<Pair> const * pairs_ref;

    Pair_Order() = default;

    Pair_Order( field<Pair> & pairs, const field<int> & n_cells, int n_cell_atoms )
            : n_pairs_per_cell_atom( field<int>( n_cell_atoms, 0 ) ),
              offset_per_cell_atom( field<int>( n_cell_atoms, 0 ) ),
              indices( std::vector<int>( pairs.size() ) ),
              pairs_ref( &pairs )
    {
        std::iota( indices.begin(), indices.end(), 0 );

        auto pair_compare_function = [&]( const int & idx_l, const int & idx_r ) -> bool {
            const Pair & l = pairs[idx_l];
            const Pair & r = pairs[idx_r];

            if( l.i < r.i )
                return true;
            else if( l.i > r.i )
                return false;

            // sub-sorting
            // if l.i == r.i, sort by the expected difference of the linear idx into the spins array
            int d_idx_l
                = l.j
                  + n_cell_atoms
                        * ( l.translations[0] + n_cells[0] * ( l.translations[1] + n_cells[1] * l.translations[2] ) );
            int d_idx_r
                = r.j
                  + n_cell_atoms
                        * ( r.translations[0] + n_cells[0] * ( r.translations[1] + n_cells[1] * r.translations[2] ) );

            return d_idx_l < d_idx_r;
        };

        std::sort( indices.begin(), indices.end(), pair_compare_function );

        // Count how many pairs interact with each cell atom
        for( const auto & p : pairs )
        {
            n_pairs_per_cell_atom[p.i]++;
        }

        // Compute the offsets at which each cell atom has to look into the sorted pairs vector
        for( int i = 1; i < n_pairs_per_cell_atom.size(); i++ )
        {
            offset_per_cell_atom[i] += offset_per_cell_atom[i - 1] + n_pairs_per_cell_atom[i - 1];
        }
    }

    // Use the indices to sort an input field
    template<typename T>
    void Sort( field<T> & v ) const
    {
        assert( v.size() == indices.size() );

        // Sort the array out of place
        field<T> v_copy = v;
        for( int i = 0; i < indices.size(); i++ )
        {
            v[i] = v_copy[indices[i]];
        }
    }

    // For debugging
    void Print( int n_pairs_print = 3 ) const
    {
        fmt::print( "== Pair Order ==\n" );

        for( auto i : indices )
        {
            fmt::print( "{} ", i );
        }
        fmt::print( "\n" );

        fmt::print( "n_pairs_total = {}\n", pairs_ref->size() );
        for( int i = 0; i < n_pairs_per_cell_atom.size(); i++ )
        {
            fmt::print( "Cell atom [{}]\n", i );
            fmt::print( "    n_pairs = {}\n", n_pairs_per_cell_atom[i] );
            fmt::print( "    offset  = {}\n", offset_per_cell_atom[i] );
            fmt::print( "    (Up to) first {} pairs:\n", n_pairs_print );

            for( int j = 0; j < std::min( n_pairs_print, n_pairs_per_cell_atom[i] ); j++ )
            {
                auto const & p = ( *pairs_ref )[offset_per_cell_atom[i] + j];
                fmt::print(
                    "      - #{} = {:^4} {:^4} {:^4} {:^4} {:^4}\n", j, p.i, p.j, p.translations[0], p.translations[1],
                    p.translations[2] );
            }
        }
    }
};

} // namespace Engine

#endif