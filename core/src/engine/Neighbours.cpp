#include <engine/Neighbours.hpp>
#include <utility/Logging.hpp>

#include <Eigen/Dense>

#include <fmt/format.h>

#include <limits>

namespace Engine
{
namespace Neighbours
{

std::vector<scalar> Get_Shell_Radii( const Data::Geometry & geometry, const std::size_t n_shells )
{
    const scalar min_shell_width = 1e-3;

    auto shell_radii = std::vector<scalar>( n_shells );

    const Vector3 ta = geometry.lattice_constant * geometry.bravais_vectors[0];
    const Vector3 tb = geometry.lattice_constant * geometry.bravais_vectors[1];
    const Vector3 tc = geometry.lattice_constant * geometry.bravais_vectors[2];

    const Matrix3 bravais_matrix = geometry.lattice_constant * geometry.bravaisMatrix();

    // The n_shells + 2 is a value that is big enough by experience to produce enough needed shells, but is small enough
    // to run sufficiently fast
    int max_n_translations = n_shells + 2;

    // Abort conditions for all 3 vectors
    const int i_max = ta.norm() == 0 ? 0 : std::min( max_n_translations, geometry.n_cells[0] - 1 );
    const int j_max = tb.norm() == 0 ? 0 : std::min( max_n_translations, geometry.n_cells[1] - 1 );
    const int k_max = tc.norm() == 0 ? 0 : std::min( max_n_translations, geometry.n_cells[2] - 1 );

    scalar previous_radius = 0, outermost_radius = 0;
    for( auto & shell_radius : shell_radii )
    {
        // scanning for the smallest possible radus in the interval: (previous_radius, outermost_radius)
        // this achieves finding the next-smallest occuring radius
        previous_radius  = outermost_radius;
        outermost_radius = std::numeric_limits<scalar>::max();
        for( int atom_one = 0; atom_one < geometry.n_cell_atoms; ++atom_one )
        {
            for( int atom_two = 0; atom_two < geometry.n_cell_atoms; ++atom_two )
            {
                const auto delta_basis
                    = bravais_matrix * ( geometry.cell_atoms[atom_two] - geometry.cell_atoms[atom_one] );
                // Note: due to symmetry we only need to check half the space
                for( int i = i_max; i >= 0; --i )
                {
                    for( int j = j_max; j >= -j_max; --j )
                    {
                        for( int k = k_max; k >= -k_max; --k )
                        {
                            if( !( atom_one == atom_two && i == 0 && j == 0 && k == 0 ) )
                            {
                                const scalar pos_delta = ( delta_basis + i * ta + j * tb + k * tc ).norm();
                                if( pos_delta < outermost_radius && pos_delta - previous_radius > min_shell_width )
                                {
                                    outermost_radius = pos_delta;
                                }
                            }
                        }
                    }
                }
            }
        }
        shell_radius = outermost_radius;
    }

    return shell_radii;
}

void Get_Neighbours_in_Shells(
    const Data::Geometry & geometry, std::size_t n_shells, pairfield & neighbours, intfield & shells,
    bool use_redundant_neighbours )
{
    const scalar min_shell_width = 1e-3;

    auto shell_radii = Get_Shell_Radii( geometry, n_shells );

    const Vector3 ta = geometry.lattice_constant * geometry.bravais_vectors[0];
    const Vector3 tb = geometry.lattice_constant * geometry.bravais_vectors[1];
    const Vector3 tc = geometry.lattice_constant * geometry.bravais_vectors[2];

    const Matrix3 bravais_matrix = geometry.lattice_constant * geometry.bravaisMatrix();

    // The n_shells + 2 is a value that is big enough by experience to produce enough needed shells, but is small enough
    // to run sufficiently fast
    const int max_n_translations = n_shells + 2;

    // Abort condidions for all 3 vectors
    const int i_max = ta.norm() == 0 ? 0 : std::min( max_n_translations, geometry.n_cells[0] - 1 );
    const int j_max = tb.norm() == 0 ? 0 : std::min( max_n_translations, geometry.n_cells[1] - 1 );
    const int k_max = tc.norm() == 0 ? 0 : std::min( max_n_translations, geometry.n_cells[2] - 1 );

    for( int atom_one = 0; atom_one < geometry.n_cell_atoms; ++atom_one )
    {
        int atom_two = 0;
        if( !use_redundant_neighbours )
            atom_two = atom_one;
        for( ; atom_two < geometry.n_cell_atoms; ++atom_two )
        {
            const auto delta_basis = bravais_matrix * ( geometry.cell_atoms[atom_two] - geometry.cell_atoms[atom_one] );
            for( int ishell = 0; ishell < n_shells; ++ishell )
            {
                const auto radius = shell_radii[ishell];
                for( int i = i_max; i >= -i_max; --i )
                {
                    for( int j = j_max; j >= -j_max; --j )
                    {
                        for( int k = k_max; k >= -k_max; --k )
                        {
                            if( ( atom_two > atom_one )
                                || ( i > 0 || ( i == 0 && j > 0 ) || ( i == 0 && j == 0 && k > 0 ) )
                                || use_redundant_neighbours )
                            {
                                const scalar pos_delta = ( delta_basis + i * ta + j * tb + k * tc ).norm();
                                if( std::abs( pos_delta - radius ) < min_shell_width )
                                {
                                    neighbours.push_back( { atom_one, atom_two, { i, j, k } } );
                                    shells.push_back( static_cast<int>( ishell ) );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

pairfield Get_Pairs_in_Radius( const Data::Geometry & geometry, scalar radius )
{
    // Check for a meaningful radius
    const scalar epsilon = 1e-6;
    if( std::abs( radius ) < epsilon )
    {
        Log( Utility::Log_Level::Error, Utility::Log_Sender::API,
             fmt::format(
                 "Generating pairs within a radius of less than {} is not supported, but you passed {}", epsilon,
                 radius ),
             -1, -1 );
        return {};
    }

    Vector3 ta = geometry.lattice_constant * geometry.bravais_vectors[0];
    Vector3 tb = geometry.lattice_constant * geometry.bravais_vectors[1];
    Vector3 tc = geometry.lattice_constant * geometry.bravais_vectors[2];

    Vector3 bounds_diff = geometry.bounds_max - geometry.bounds_min;
    Vector3 ratio       = {
        bounds_diff[0] / std::max( 1, geometry.n_cells[0] ),
        bounds_diff[1] / std::max( 1, geometry.n_cells[1] ),
        bounds_diff[2] / std::max( 1, geometry.n_cells[2] ),
    };

    // This should give enough translations to contain all DDI pairs
    int imax = 0, jmax = 0, kmax = 0;

    // If radius < 0 we take all pairs
    if( radius > 0 )
    {
        if( bounds_diff[0] > 0 )
            imax = std::min(
                geometry.n_cells[0] - 1, static_cast<int>( 1.1 * radius * geometry.n_cells[0] / bounds_diff[0] ) );
        if( bounds_diff[1] > 0 )
            jmax = std::min(
                geometry.n_cells[1] - 1, static_cast<int>( 1.1 * radius * geometry.n_cells[1] / bounds_diff[1] ) );
        if( bounds_diff[2] > 0 )
            kmax = std::min(
                geometry.n_cells[2] - 1, static_cast<int>( 1.1 * radius * geometry.n_cells[2] / bounds_diff[2] ) );
    }
    else
    {
        imax = geometry.n_cells[0] - 1;
        jmax = geometry.n_cells[1] - 1;
        kmax = geometry.n_cells[2] - 1;
    }

    // Abort conditions for all 3 vectors
    if( ta.norm() == 0.0 )
        imax = 0;
    if( tb.norm() == 0.0 )
        jmax = 0;
    if( tc.norm() == 0.0 )
        kmax = 0;

    auto pairs = pairfield( 0 );

    int i = 0, j = 0, k = 0;
    scalar pos_delta   = 0;
    Vector3 position_i = { 0, 0, 0 };
    Vector3 position_j = { 0, 0, 0 };

    for( int iatom = 0; iatom < geometry.n_cell_atoms; ++iatom )
    {
        position_i = geometry.positions[iatom];

        for( i = -imax; i <= imax; ++i )
        {
            for( j = -jmax; j <= jmax; ++j )
            {
                for( k = -kmax; k <= kmax; ++k )
                {
                    for( int jatom = 0; jatom < geometry.n_cell_atoms; ++jatom )
                    {
                        position_j = geometry.positions[jatom] + i * ta + j * tb + k * tc;
                        pos_delta  = ( position_i - position_j ).norm();
                        if( pos_delta < radius
                            && pos_delta > std::numeric_limits<scalar>::epsilon() ) // Exclude self-interactions
                        {
                            pairs.push_back( { iatom, jatom, { i, j, k } } );
                        }
                    }
                }
            }
        }
    }

    return pairs;
}

Vector3 DMI_Normal_from_Pair( const Data::Geometry & geometry, const Pair & pair, std::int8_t chirality )
{
    Vector3 ta = geometry.lattice_constant * geometry.bravais_vectors[0];
    Vector3 tb = geometry.lattice_constant * geometry.bravais_vectors[1];
    Vector3 tc = geometry.lattice_constant * geometry.bravais_vectors[2];

    int da = pair.translations[0];
    int db = pair.translations[1];
    int dc = pair.translations[2];

    Vector3 ipos = geometry.positions[pair.i];
    Vector3 jpos = geometry.positions[pair.j] + da * ta + db * tb + dc * tc;

    if( chirality == 1 )
    {
        // Bloch chirality
        return ( jpos - ipos ).normalized();
    }
    else if( chirality == -1 )
    {
        // Inverse Bloch chirality
        return ( ipos - jpos ).normalized();
    }
    else if( chirality == 2 )
    {
        // Neel chirality (surface)
        return ( jpos - ipos ).normalized().cross( Vector3{ 0, 0, 1 } );
    }
    else if( chirality == -2 )
    {
        // Inverse Neel chirality (surface)
        return Vector3{ 0, 0, 1 }.cross( ( jpos - ipos ).normalized() );
    }
    else
    {
        return Vector3{ 0, 0, 0 };
    }
}

void DDI_from_Pair( const Data::Geometry & geometry, const Pair & pair, scalar & magnitude, Vector3 & normal )
{
    Vector3 ta = geometry.lattice_constant * geometry.bravais_vectors[0];
    Vector3 tb = geometry.lattice_constant * geometry.bravais_vectors[1];
    Vector3 tc = geometry.lattice_constant * geometry.bravais_vectors[2];

    int da = pair.translations[0];
    int db = pair.translations[1];
    int dc = pair.translations[2];

    Vector3 ipos = geometry.positions[pair.i];
    Vector3 jpos = geometry.positions[pair.j] + da * ta + db * tb + dc * tc;

    // Calculate positions and difference vector
    Vector3 vector_ij = jpos - ipos;

    // Length of difference vector
    magnitude = vector_ij.norm();
    normal    = vector_ij.normalized();
}

} // end Namespace Neighbours
} // end Namespace Engine
