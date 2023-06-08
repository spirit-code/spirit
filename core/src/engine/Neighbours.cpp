#include <engine/Neighbours.hpp>
#include <utility/Logging.hpp>

#include <Eigen/Dense>

#include <fmt/format.h>

#include <limits>

namespace Spirit::Engine::Neighbours
{

std::vector<scalar> Get_Shell_Radii( const Data::Geometry & geometry, const std::size_t n_shells )
{
    const scalar min_shell_width = 1e-3;

    auto shell_radii = std::vector<scalar>( n_shells );

    Vector3 ta = geometry.lattice_constant * geometry.bravais_vectors[0];
    Vector3 tb = geometry.lattice_constant * geometry.bravais_vectors[1];
    Vector3 tc = geometry.lattice_constant * geometry.bravais_vectors[2];

    // The n_shells + 2 is a value that is big enough by experience to produce enough needed shells, but is small enough
    // to run sufficiently fast
    int max_n_translations = n_shells + 2;

    int i_max = std::min( max_n_translations, geometry.n_cells[0] - 1 );
    int j_max = std::min( max_n_translations, geometry.n_cells[1] - 1 );
    int k_max = std::min( max_n_translations, geometry.n_cells[2] - 1 );

    // Abort condidions for all 3 vectors
    if( ta.norm() == 0.0 )
        i_max = 0;
    if( tb.norm() == 0.0 )
        j_max = 0;
    if( tc.norm() == 0.0 )
        k_max = 0;

    int atom_one{ 0 }, atom_two{ 0 };
    int i{ 0 }, j{ 0 }, k{ 0 };
    scalar outermost_radius = 0, pos_delta = 0, previous_radius = 0;
    Vector3 pos_one = { 0, 0, 0 }, pos_two = { 0, 0, 0 };
    for( auto & shell_radius : shell_radii )
    {
        previous_radius = outermost_radius;
        // Starting from the maximum representable value, determine the smallest shell that is more than min_shell_width
        // wider than the previous
        outermost_radius = std::numeric_limits<scalar>::max();
        for( atom_one = 0; atom_one < geometry.n_cell_atoms; ++atom_one )
        {
            pos_one = geometry.cell_atoms[atom_one];
            // Note: due to symmetry we only need to check half the space
            for( i = i_max; i >= 0; --i )
            {
                for( j = j_max; j >= -j_max; --j )
                {
                    for( k = k_max; k >= -k_max; --k )
                    {
                        for( atom_two = 0; atom_two < geometry.n_cell_atoms; ++atom_two )
                        {
                            if( !( atom_one == atom_two && i == 0 && j == 0 && k == 0 ) )
                            {
                                pos_two   = geometry.cell_atoms[atom_two] + i * ta + j * tb + k * tc;
                                pos_delta = ( pos_one - pos_two ).norm();
                                if( pos_delta - previous_radius > min_shell_width && pos_delta < outermost_radius )
                                {
                                    outermost_radius = pos_delta;
                                    shell_radius     = pos_delta;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return shell_radii;
}

void Get_Neighbours_in_Shells(
    const Data::Geometry & geometry, std::size_t n_shells, pairfield & neighbours, intfield & shells,
    bool use_redundant_neighbours )
{
    const scalar min_shell_width = 1e-3;

    auto shell_radii = Get_Shell_Radii( geometry, n_shells );

    Vector3 ta = geometry.lattice_constant * geometry.bravais_vectors[0];
    Vector3 tb = geometry.lattice_constant * geometry.bravais_vectors[1];
    Vector3 tc = geometry.lattice_constant * geometry.bravais_vectors[2];

    // The n_shells + 2 is a value that is big enough by experience to produce enough needed shells, but is small enough
    // to run sufficiently fast
    int max_n_translations = n_shells + 2;

    int i_max = std::min( max_n_translations, geometry.n_cells[0] - 1 );
    int j_max = std::min( max_n_translations, geometry.n_cells[1] - 1 );
    int k_max = std::min( max_n_translations, geometry.n_cells[2] - 1 );

    // If redundant neighbours should not be used, we restrict the search to half of the space
    int i_min = -i_max;
    int j_min = -j_max;
    int k_min = -k_max;

    // Abort condidions for all 3 vectors
    if( ta.norm() == 0.0 )
        i_max = 0;
    if( tb.norm() == 0.0 )
        j_max = 0;
    if( tc.norm() == 0.0 )
        k_max = 0;

    int second_atom_min = 0;
    int atom_one{ 0 }, atom_two{ 0 }, i{ 0 }, j{ 0 }, k{ 0 };
    std::size_t ishell = 0;
    scalar pos_delta = 0, radius = 0;
    Vector3 pos_one = { 0, 0, 0 }, pos_two = { 0, 0, 0 };
    for( atom_one = 0; atom_one < geometry.n_cell_atoms; ++atom_one )
    {
        if( !use_redundant_neighbours )
            second_atom_min = atom_one;

        pos_one = geometry.cell_atoms[atom_one];
        for( ishell = 0; ishell < n_shells; ++ishell )
        {
            radius = shell_radii[ishell];
            for( i = i_max; i >= i_min; --i )
            {
                for( j = j_max; j >= j_min; --j )
                {
                    for( k = k_max; k >= k_min; --k )
                    {
                        for( atom_two = second_atom_min; atom_two < geometry.n_cell_atoms; ++atom_two )
                        {
                            if( ( atom_two > atom_one )
                                || ( i > 0 || ( i == 0 && j > 0 ) || ( i == 0 && j == 0 && k > 0 ) )
                                || use_redundant_neighbours )
                            {
                                pos_two   = geometry.cell_atoms[atom_two] + i * ta + j * tb + k * tc;
                                pos_delta = ( pos_one - pos_two ).norm();
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

} // namespace Spirit::Engine::Neighbours
