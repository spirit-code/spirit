#pragma once
#ifndef SPIRIT_CORE_ENGINE_INDEXING_HPP
#define SPIRIT_CORE_ENGINE_INDEXING_HPP

#include <engine/Vectormath_Defines.hpp>

namespace Engine
{
namespace Indexing
{

// Note: translations must lie within bounds of n_cells
inline int
idx_from_translations( const intfield & n_cells, const int n_cell_atoms, const std::array<int, 3> & translations )
{
    const auto & Na = n_cells[0];
    const auto & Nb = n_cells[1];
    const auto & Nc = n_cells[2];
    const auto & N  = n_cell_atoms;

    const auto & da = translations[0];
    const auto & db = translations[1];
    const auto & dc = translations[2];

    return da * N + db * N * Na + dc * N * Na * Nb;
}

#ifndef SPIRIT_USE_CUDA

// Get the linear index in a n-D array where tupel contains the components in n-dimensions from fatest to slowest
// varying and maxVal is the extent in every dimension
inline int idx_from_tupel( const intfield & tupel, const intfield & maxVal )
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
inline void tupel_from_idx( const int & idx, intfield & tupel, const intfield & maxVal )
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

inline int idx_from_translations(
    const intfield & n_cells, const int n_cell_atoms, const std::array<int, 3> & translations_i,
    const std::array<int, 3> & translations )
{
    const auto & Na = n_cells[0];
    const auto & Nb = n_cells[1];
    const auto & Nc = n_cells[2];
    const auto & N  = n_cell_atoms;

    int da = translations_i[0] + translations[0];
    int db = translations_i[1] + translations[1];
    int dc = translations_i[2] + translations[2];

    if( translations[0] < 0 )
        da += N * Na;
    if( translations[1] < 0 )
        db += N * Na * Nb;
    if( translations[2] < 0 )
        dc += N * Na * Nb * Nc;

    return ( da % Na ) * N + ( db % Nb ) * N * Na + ( dc % Nc ) * N * Na * Nb;
}

inline bool boundary_conditions_fulfilled(
    const intfield & n_cells, const intfield & boundary_conditions, const std::array<int, 3> & translations_i,
    const std::array<int, 3> & translations_j )
{
    int da = translations_i[0] + translations_j[0];
    int db = translations_i[1] + translations_j[1];
    int dc = translations_i[2] + translations_j[2];
    return (
        ( boundary_conditions[0] || ( 0 <= da && da < n_cells[0] ) )
        && ( boundary_conditions[1] || ( 0 <= db && db < n_cells[1] ) )
        && ( boundary_conditions[2] || ( 0 <= dc && dc < n_cells[2] ) ) );
}

#endif
#ifdef SPIRIT_USE_CUDA

// // Get the linear index in a n-D array where tupel contains the components in n-dimensions from fatest to slowest
// // varying and maxVal is the extent in every dimension
// inline __device__ int cu_idx_from_tupel( intfield & tupel, intfield & maxVal )
// {
//     int idx  = 0;
//     int mult = 1;
//     for( int i = 0; i < tupel.size(); i++ )
//     {
//         idx += mult * tupel[i];
//         mult *= maxVal[i];
//     }
//     return idx;
// }

// reverse of idx_from_tupel
inline __device__ void cu_tupel_from_idx( const int & idx, int * tupel, const int * maxVal, const int n )
{
    int idx_diff = idx;
    int div      = 1;
    for( int i = 0; i < n - 1; i++ )
        div *= maxVal[i];
    for( int i = n - 1; i > 0; i-- )
    {
        tupel[i] = idx_diff / div;
        idx_diff -= tupel[i] * div;
        div /= maxVal[i - 1];
    }
    tupel[0] = idx_diff / div;
}

inline void tupel_from_idx( const int & idx, intfield & tupel, const intfield & maxVal )
{
    int idx_diff = idx;
    int div      = 1;
    for( int i = 0; i < maxVal.size() - 1; i++ )
        div *= maxVal[i];
    for( int i = maxVal.size() - 1; i > 0; i-- )
    {
        tupel[i] = idx_diff / div;
        idx_diff -= tupel[i] * div;
        div /= maxVal[i - 1];
    }
    tupel[0] = idx_diff / div;
}

inline int idx_from_translations(
    const intfield & n_cells, const int n_cell_atoms, const std::array<int, 3> & translations_i,
    const std::array<int, 3> & translations )
{
    const int Na = n_cells[0];
    const int Nb = n_cells[1];
    const int Nc = n_cells[2];
    const int N  = n_cell_atoms;

    int da = translations_i[0] + translations[0];
    int db = translations_i[1] + translations[1];
    int dc = translations_i[2] + translations[2];

    if( translations[0] < 0 )
        da += N * Na;
    if( translations[1] < 0 )
        db += N * Na * Nb;
    if( translations[2] < 0 )
        dc += N * Na * Nb * Nc;

    int idx = ( da % Na ) * N + ( db % Nb ) * N * Na + ( dc % Nc ) * N * Na * Nb;

    return idx;
}

inline bool boundary_conditions_fulfilled(
    const intfield & n_cells, const intfield & boundary_conditions, const std::array<int, 3> & translations_i,
    const std::array<int, 3> & translations_j )
{
    const int da = translations_i[0] + translations_j[0];
    const int db = translations_i[1] + translations_j[1];
    const int dc = translations_i[2] + translations_j[2];
    return (
        ( boundary_conditions[0] || ( 0 <= da && da < n_cells[0] ) )
        && ( boundary_conditions[1] || ( 0 <= db && db < n_cells[1] ) )
        && ( boundary_conditions[2] || ( 0 <= dc && dc < n_cells[2] ) ) );
}

__inline__ __device__ bool cu_check_atom_type( int atom_type )
{
#ifdef SPIRIT_ENABLE_DEFECTS
    // If defects are enabled we check for
    //		vacancies (type < 0)
    if( atom_type >= 0 )
        return true;
    else
        return false;
#else
    // Else we just return true
    return true;
#endif
}

__inline__ __device__ bool cu_check_atom_type( const int atom_type, const int reference_type )
{
#ifdef SPIRIT_ENABLE_DEFECTS
    // If defects are enabled we do a check if
    //		atom types match.
    if( atom_type == reference_type )
        return true;
    else
        return false;
#else
    // Else we just return true
    return true;
#endif
}

// Calculates, for a spin i, a pair spin's index j.
// This function takes into account boundary conditions and atom types and returns `-1` if any condition is not met.
__inline__ __device__ int cu_idx_from_pair(
    int ispin, const int * boundary_conditions, const int * n_cells, int N, const int * atom_types, const Pair & pair,
    bool invert = false )
{
    // Invalid index if atom type of spin i is not correct
    if( pair.i != ispin % N || !cu_check_atom_type( atom_types[ispin] ) )
        return -1;

    // Number of cells
    const auto & Na = n_cells[0];
    const auto & Nb = n_cells[1];
    const auto & Nc = n_cells[2];

    // Invalid index if translations reach out over the lattice bounds
    if( std::abs( pair.translations[0] ) > Na || std::abs( pair.translations[1] ) > Nb
        || std::abs( pair.translations[2] ) > Nc )
        return -1;

    // Translations (cell) of spin i
    int nic = ispin / ( N * Na * Nb );
    int nib = ( ispin - nic * N * Na * Nb ) / ( N * Na );
    int nia = ( ispin - nic * N * Na * Nb - nib * N * Na ) / N;

    // Translations (cell) of spin j (possibly outside of non-periodical domain)
    int pm = 1;
    if( invert )
        pm = -1;
    int nja = nia + pm * pair.translations[0];
    int njb = nib + pm * pair.translations[1];
    int njc = nic + pm * pair.translations[2];

    // Check boundary conditions: a
    if( boundary_conditions[0] || ( 0 <= nja && nja < Na ) )
    {
        // Boundary conditions fulfilled
        // Find the translations of spin j within the non-periodical domain
        if( nja < 0 )
            nja += Na;
        // Calculate the correct index
        if( nja >= Na )
            nja -= Na;
    }
    else
    {
        // Boundary conditions not fulfilled
        return -1;
    }

    // Check boundary conditions: b
    if( boundary_conditions[1] || ( 0 <= njb && njb < Nb ) )
    {
        // Boundary conditions fulfilled
        // Find the translations of spin j within the non-periodical domain
        if( njb < 0 )
            njb += Nb;
        // Calculate the correct index
        if( njb >= Nb )
            njb -= Nb;
    }
    else
    {
        // Boundary conditions not fulfilled
        return -1;
    }

    // Check boundary conditions: c
    if( boundary_conditions[2] || ( 0 <= njc && njc < Nc ) )
    {
        // Boundary conditions fulfilled
        // Find the translations of spin j within the non-periodical domain
        if( njc < 0 )
            njc += Nc;
        // Calculate the correct index
        if( njc >= Nc )
            njc -= Nc;
    }
    else
    {
        // Boundary conditions not fulfilled
        return -1;
    }

    // Calculate the index of spin j according to it's translations
    int jspin = pair.j + (nja)*N + (njb)*N * Na + (njc)*N * Na * Nb;

    if( !cu_check_atom_type( atom_types[jspin] ) )
        return -1;

    // Return a valid index
    return jspin;
}

#endif

inline std::array<int, 3> translations_from_idx( const intfield & n_cells, const int n_cell_atoms, int idx )
{
    std::array<int, 3> ret;
    int Na = n_cells[0];
    int Nb = n_cells[1];
    int Nc = n_cells[2];
    int N  = n_cell_atoms;

    ret[2] = idx / ( N * Na * Nb );
    ret[1] = ( idx - ret[2] * N * Na * Nb ) / ( N * Na );
    ret[0] = ( idx - ret[2] * N * Na * Nb - ret[1] * N * Na ) / N;
    return ret;
}

// Check atom types
inline bool check_atom_type( int atom_type )
{
#ifdef SPIRIT_ENABLE_DEFECTS
    // If defects are enabled we check for
    //		vacancies (type < 0)
    if( atom_type >= 0 )
        return true;
    else
        return false;
#else
    // Else we just return true
    return true;
#endif
}
inline bool check_atom_type( int atom_type, int reference_type )
{
#ifdef SPIRIT_ENABLE_DEFECTS
    // If defects are enabled we do a check if
    //		atom types match.
    if( atom_type == reference_type )
        return true;
    else
        return false;
#else
    // Else we just return true
    return true;
#endif
}

// Calculates, for a spin i, a pair spin's index j.
// This function takes into account boundary conditions and atom types and returns `-1` if any condition is not met.
inline int idx_from_pair(
    const int ispin, const intfield & boundary_conditions, const intfield & n_cells, int N, const intfield & atom_types,
    const Pair & pair, bool invert = false )
{
    // Invalid index if atom type of spin i is not correct
    if( pair.i != ispin % N || !check_atom_type( atom_types[ispin] ) )
        return -1;

    // Number of cells
    const auto & Na = n_cells[0];
    const auto & Nb = n_cells[1];
    const auto & Nc = n_cells[2];

    // Invalid index if translations reach out over the lattice bounds
    if( std::abs( pair.translations[0] ) > Na || std::abs( pair.translations[1] ) > Nb
        || std::abs( pair.translations[2] ) > Nc )
        return -1;

    // Translations (cell) of spin i
    int nic = ispin / ( N * Na * Nb );
    int nib = ( ispin - nic * N * Na * Nb ) / ( N * Na );
    int nia = ( ispin - nic * N * Na * Nb - nib * N * Na ) / N;

    int pm = 1;
    if( invert )
        pm = -1;
    // Translations (cell) of spin j (possibly outside of non-periodical domain)
    int nja = nia + pm * pair.translations[0];
    int njb = nib + pm * pair.translations[1];
    int njc = nic + pm * pair.translations[2];

    // Check boundary conditions: a
    if( boundary_conditions[0] || ( 0 <= nja && nja < Na ) )
    {
        // Boundary conditions fulfilled
        // Find the translations of spin j within the non-periodical domain
        if( nja < 0 )
            nja += Na;
        // Calculate the correct index
        if( nja >= Na )
            nja -= Na;
    }
    else
    {
        // Boundary conditions not fulfilled
        return -1;
    }

    // Check boundary conditions: b
    if( boundary_conditions[1] || ( 0 <= njb && njb < Nb ) )
    {
        // Boundary conditions fulfilled
        // Find the translations of spin j within the non-periodical domain
        if( njb < 0 )
            njb += Nb;
        // Calculate the correct index
        if( njb >= Nb )
            njb -= Nb;
    }
    else
    {
        // Boundary conditions not fulfilled
        return -1;
    }

    // Check boundary conditions: c
    if( boundary_conditions[2] || ( 0 <= njc && njc < Nc ) )
    {
        // Boundary conditions fulfilled
        // Find the translations of spin j within the non-periodical domain
        if( njc < 0 )
            njc += Nc;
        // Calculate the correct index
        if( njc >= Nc )
            njc -= Nc;
    }
    else
    {
        // Boundary conditions not fulfilled
        return -1;
    }

    // Calculate the index of spin j according to it's translations
    int jspin = pair.j + (nja)*N + (njb)*N * Na + (njc)*N * Na * Nb;

    // Invalid index if atom type of spin j is not correct
    if( !check_atom_type( atom_types[jspin] ) )
        return -1;

    // Return a valid index
    return jspin;
}

// Re-distribute a given field according to a new set of dimensions.
template<typename T>
field<T> change_dimensions(
    field<T> & oldfield, const int n_cell_atoms_old, const intfield & n_cells_old, const int n_cell_atoms_new,
    const intfield & n_cells_new, T default_value, std::array<int, 3> shift = std::array<int, 3>{ 0, 0, 0 } )
{
    // As a workaround for compatibility with the intel compiler, the loop boundaries are copied to a local array;
    //  not sure whether the "parallel loops with collapse must be perfectly nested" error (without this) is a compiler
    //  bug or standard conform behaviour
    const int n_cells_new_local_copy[] = { n_cells_new[0], n_cells_new[1], n_cells_new[2] };

    int N_new = n_cell_atoms_new * n_cells_new[0] * n_cells_new[1] * n_cells_new[2];
    field<T> newfield( N_new, default_value );

#pragma omp parallel for collapse( 3 )
    for( int i = 0; i < n_cells_new_local_copy[0]; ++i )
    {
        for( int j = 0; j < n_cells_new_local_copy[1]; ++j )
        {
            for( int k = 0; k < n_cells_new_local_copy[2]; ++k )
            {
                for( int iatom = 0; iatom < n_cell_atoms_new; ++iatom )
                {
                    int idx_new = iatom + idx_from_translations( n_cells_new, n_cell_atoms_new, { i, j, k }, shift );

                    if( ( iatom < n_cell_atoms_old ) && ( i < n_cells_old[0] ) && ( j < n_cells_old[1] )
                        && ( k < n_cells_old[2] ) )
                    {
                        int idx_old       = iatom + idx_from_translations( n_cells_old, n_cell_atoms_old, { i, j, k } );
                        newfield[idx_new] = oldfield[idx_old];
                    }
                    // else
                    //     newfield[idx_new] = default_value;
                }
            }
        }
    }
    return newfield;
}

} // namespace Indexing
} // namespace Engine

#endif
