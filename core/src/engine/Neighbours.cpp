#include <engine/Neighbours.hpp>

#include <Eigen/Dense>

namespace Engine
{
namespace Neighbours
{

std::vector<scalar> Get_Shell_Radius( const Data::Geometry & geometry, const int n_shells )
{
    const scalar shell_width = 1e-3;
    auto shell_radius        = std::vector<scalar>( n_shells );

    Vector3 ta = geometry.lattice_constant * geometry.bravais_vectors[0];
    Vector3 tb = geometry.lattice_constant * geometry.bravais_vectors[1];
    Vector3 tc = geometry.lattice_constant * geometry.bravais_vectors[2];

    // The n_shells + 2 is a value that is big enough by experience to
    // produce enough needed shells, but is small enough to run sufficiently fast
    int tMax = n_shells + 2;
    int imax = std::min( tMax, geometry.n_cells[0] - 1 ), jmax = std::min( tMax, geometry.n_cells[1] - 1 ),
        kmax = std::min( tMax, geometry.n_cells[2] - 1 );

    // Abort condidions for all 3 vectors
    if( ta.norm() == 0.0 )
        imax = 0;
    if( tb.norm() == 0.0 )
        jmax = 0;
    if( tc.norm() == 0.0 )
        kmax = 0;

    int i, j, k, iatom, jatom, ishell;
    scalar current_radius = 0, dx, min_distance = 0;
    Vector3 x0 = { 0, 0, 0 }, x1 = { 0, 0, 0 };
    for( ishell = 0; ishell < n_shells; ++ishell )
    {
        min_distance   = current_radius;
        current_radius = 1e10;
        for( iatom = 0; iatom < geometry.n_cell_atoms; ++iatom )
        {
            x0 = geometry.positions[iatom];
            // Note: due to symmetry we only need to check half the space
            for( i = imax; i >= 0; --i )
            {
                for( j = jmax; j >= -jmax; --j )
                {
                    for( k = kmax; k >= -kmax; --k )
                    {
                        for( jatom = 0; jatom < geometry.n_cell_atoms; ++jatom )
                        {
                            if( !( iatom == jatom && i == 0 && j == 0 && k == 0 ) )
                            {
                                x1 = geometry.positions[jatom] + i * ta + j * tb + k * tc;
                                dx = ( x0 - x1 ).norm();
                                if( dx - min_distance > shell_width && dx < current_radius )
                                {
                                    current_radius       = dx;
                                    shell_radius[ishell] = dx;
                                }
                            }
                        } // endfor jatom
                    }     // endfor k
                }         // endfor j
            }             // endfor i
        }                 // endfor iatom
    }

    return shell_radius;
}

void Get_Neighbours_in_Shells(
    const Data::Geometry & geometry, int n_shells, pairfield & neighbours, intfield & shells,
    bool use_redundant_neighbours )
{
    const scalar shell_width = 1e-3;
    auto shell_radius        = Get_Shell_Radius( geometry, n_shells );

    Vector3 ta = geometry.lattice_constant * geometry.bravais_vectors[0];
    Vector3 tb = geometry.lattice_constant * geometry.bravais_vectors[1];
    Vector3 tc = geometry.lattice_constant * geometry.bravais_vectors[2];

    // The n_shells + 2 is a value that is big enough by experience to
    // produce enough needed shells, but is small enough to run sufficiently fast
    int tMax = n_shells + 2;
    int imax = std::min( tMax, geometry.n_cells[0] - 1 ), jmax = std::min( tMax, geometry.n_cells[1] - 1 ),
        kmax = std::min( tMax, geometry.n_cells[2] - 1 );
    int imin, jmin, kmin, jatommin;
    // If redundant neighbours should not be used, we restrict the search to half of the space
    imin = -imax;
    jmin = -jmax;
    kmin = -kmax;

    // Abort condidions for all 3 vectors
    if( ta.norm() == 0.0 )
        imax = 0;
    if( tb.norm() == 0.0 )
        jmax = 0;
    if( tc.norm() == 0.0 )
        kmax = 0;

    int i, j, k, iatom, jatom, ishell;
    scalar dx, radius;
    Vector3 x0 = { 0, 0, 0 }, x1 = { 0, 0, 0 };
    for( iatom = 0; iatom < geometry.n_cell_atoms; ++iatom )
    {
        if( use_redundant_neighbours )
            jatommin = 0;
        else
            jatommin = iatom;

        x0 = geometry.positions[iatom];
        for( ishell = 0; ishell < n_shells; ++ishell )
        {
            radius = shell_radius[ishell];
            for( i = imax; i >= imin; --i )
            {
                for( j = jmax; j >= jmin; --j )
                {
                    for( k = kmax; k >= kmin; --k )
                    {
                        for( jatom = jatommin; jatom < geometry.n_cell_atoms; ++jatom )
                        {
                            if( ( jatom > iatom ) || ( i > 0 || ( i == 0 && j > 0 ) || ( i == 0 && j == 0 && k > 0 ) )
                                || use_redundant_neighbours )
                            {
                                x1 = geometry.positions[jatom] + i * ta + j * tb + k * tc;
                                dx = ( x0 - x1 ).norm();
                                if( std::abs( dx - radius ) < shell_width )
                                {
                                    Pair neigh;
                                    neigh.i               = iatom;
                                    neigh.j               = jatom;
                                    neigh.translations[0] = i;
                                    neigh.translations[1] = j;
                                    neigh.translations[2] = k;
                                    neighbours.push_back( neigh );
                                    shells.push_back( ishell );
                                }
                            }
                        } // endfor jatom
                    }     // endfor k
                }         // endfor j
            }             // endfor i
        }                 // endfor ishell
    }                     // endfor iatom
}

pairfield Get_Pairs_in_Radius( const Data::Geometry & geometry, scalar radius )
{
    auto pairs = pairfield( 0 );

    // Check for a meaningful radius
    if( std::abs( radius ) > 1e-6 )
    {
        Vector3 ta = geometry.lattice_constant * geometry.bravais_vectors[0];
        Vector3 tb = geometry.lattice_constant * geometry.bravais_vectors[1];
        Vector3 tc = geometry.lattice_constant * geometry.bravais_vectors[2];

        Vector3 bounds_diff = geometry.bounds_max - geometry.bounds_min;
        Vector3 ratio       = { bounds_diff[0] / std::max( 1, geometry.n_cells[0] ),
                          bounds_diff[1] / std::max( 1, geometry.n_cells[1] ),
                          bounds_diff[2] / std::max( 1, geometry.n_cells[2] ) };

        // This should give enough translations to contain all DDI pairs
        int imax = 0, jmax = 0, kmax = 0;

        // If radius < 0 we take all pairs
        if( radius > 0 )
        {
            if( bounds_diff[0] > 0 )
                imax
                    = std::min( geometry.n_cells[0] - 1, (int)( 1.1 * radius * geometry.n_cells[0] / bounds_diff[0] ) );
            if( bounds_diff[1] > 0 )
                jmax
                    = std::min( geometry.n_cells[1] - 1, (int)( 1.1 * radius * geometry.n_cells[1] / bounds_diff[1] ) );
            if( bounds_diff[2] > 0 )
                kmax
                    = std::min( geometry.n_cells[2] - 1, (int)( 1.1 * radius * geometry.n_cells[2] / bounds_diff[2] ) );
        }
        else
        {
            imax = geometry.n_cells[0] - 1;
            jmax = geometry.n_cells[1] - 1;
            kmax = geometry.n_cells[2] - 1;
        }

        int i, j, k;
        scalar dx;
        Vector3 x0 = { 0, 0, 0 }, x1 = { 0, 0, 0 };

        // Abort conditions for all 3 vectors
        if( ta.norm() == 0.0 )
            imax = 0;
        if( tb.norm() == 0.0 )
            jmax = 0;
        if( tc.norm() == 0.0 )
            kmax = 0;

        for( int iatom = 0; iatom < geometry.n_cell_atoms; ++iatom )
        {
            x0 = geometry.positions[iatom];

            for( i = -imax; i <= imax; ++i )
            {
                for( j = -jmax; j <= jmax; ++j )
                {
                    for( k = -kmax; k <= kmax; ++k )
                    {
                        for( int jatom = 0; jatom < geometry.n_cell_atoms; ++jatom )
                        {
                            x1 = geometry.positions[jatom] + i * ta + j * tb + k * tc;
                            dx = ( x0 - x1 ).norm();
                            if( dx < radius && dx > 1e-8 ) // Exclude self-interactions
                            {
                                pairs.push_back( { iatom, jatom, { i, j, k } } );
                            }
                        } // endfor jatom
                    }     // endfor k
                }         // endfor j
            }             // endfor i
        }                 // endfor iatom
    }

    return pairs;
}

Vector3 DMI_Normal_from_Pair( const Data::Geometry & geometry, const Pair & pair, int chirality )
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
