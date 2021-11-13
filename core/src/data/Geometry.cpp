#include <data/Geometry.hpp>
#include <engine/Neighbours.hpp>
#include <engine/Vectormath.hpp>
#include <utility/Exception.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "Qhull.h"
#include "QhullFacetList.h"
#include "QhullVertexSet.h"

#include <fmt/ostream.h>

#include <array>
#include <cstdint>
#include <random>

namespace Data
{

Geometry::Geometry(
    std::vector<Vector3> bravais_vectors, intfield n_cells, std::vector<Vector3> cell_atoms,
    Basis_Cell_Composition cell_composition, scalar lattice_constant, Pinning pinning, Defects defects )
        : bravais_vectors( bravais_vectors ),
          n_cells( n_cells ),
          n_cell_atoms( cell_atoms.size() ),
          cell_atoms( cell_atoms ),
          cell_composition( cell_composition ),
          lattice_constant( lattice_constant ),
          nos( cell_atoms.size() * n_cells[0] * n_cells[1] * n_cells[2] ),
          nos_nonvacant( cell_atoms.size() * n_cells[0] * n_cells[1] * n_cells[2] ),
          n_cells_total( n_cells[0] * n_cells[1] * n_cells[2] ),
          pinning( pinning ),
          defects( defects )
{
    // Generate positions and atom types
    this->positions = vectorfield( this->nos );
    this->generatePositions();

    // Calculate some useful info
    this->calculateBounds();
    this->calculateUnitCellBounds();
    this->calculateDimensionality();

    // Calculate center of the System
    this->center = 0.5 * ( this->bounds_min + this->bounds_max );

    // Generate default atom_types, mu_s and pinning masks
    this->atom_types        = intfield( this->nos, 0 );
    this->mu_s              = scalarfield( this->nos, 1 );
    this->mask_unpinned     = intfield( this->nos, 1 );
    this->mask_pinned_cells = vectorfield( this->nos, { 0, 0, 0 } );

    // Set atom types, mu_s
    this->applyCellComposition();

    // Apply additional pinned sites
    for( std::size_t isite = 0; isite < pinning.sites.size(); ++isite )
    {
        const auto & site = pinning.sites[isite];
        std::size_t ispin = site.i
                            + Engine::Vectormath::idx_from_translations(
                                this->n_cells, this->n_cell_atoms,
                                { site.translations[0], site.translations[1], site.translations[2] } );

        this->mask_unpinned[ispin]     = 0;
        this->mask_pinned_cells[ispin] = pinning.spins[isite];
    }

    // Apply additional defect sites
    for( std::size_t i = 0; i < defects.sites.size(); ++i )
    {
        auto & defect     = defects.sites[i];
        std::size_t ispin = defects.sites[i].i
                            + Engine::Vectormath::idx_from_translations(
                                this->n_cells, this->n_cell_atoms,
                                { defect.translations[0], defect.translations[1], defect.translations[2] } );
        this->atom_types[ispin] = defects.types[i];
        this->mu_s[ispin]       = 0.0;
    }

    // Calculate the type of geometry
    this->calculateGeometryType();

    // For updates of triangulation and tetrahedra
    this->last_update_n_cell_step = -1;
    this->last_update_n_cells     = intfield( 3, -1 );
}

void Geometry::generatePositions()
{
    const scalar epsilon = 1e-6;

    // Check for erronous input placing two spins on the same location
    std::int64_t max_a = std::min( 10, n_cells[0] );
    std::int64_t max_b = std::min( 10, n_cells[1] );
    std::int64_t max_c = std::min( 10, n_cells[2] );
    Vector3 diff;
    for( std::int64_t i = 0; i < n_cell_atoms; ++i )
    {
        for( std::int64_t j = 0; j < n_cell_atoms; ++j )
        {
            for( std::int64_t da = -max_a; da <= max_a; ++da )
            {
                for( std::int64_t db = -max_b; db <= max_b; ++db )
                {
                    for( std::int64_t dc = -max_c; dc <= max_c; ++dc )
                    {
                        // Check if translated basis atom is at position of another basis atom
                        diff = cell_atoms[i] - ( cell_atoms[j] + Vector3{ scalar( da ), scalar( db ), scalar( dc ) } );

                        bool same_position = std::abs( diff[0] ) < epsilon && std::abs( diff[1] ) < epsilon
                                             && std::abs( diff[2] ) < epsilon;

                        if( same_position && ( i != j || da != 0 || db != 0 || dc != 0 ) )
                        {
                            Vector3 position
                                = lattice_constant
                                  * ( ( static_cast<scalar>( da ) + cell_atoms[i][0] ) * bravais_vectors[0]
                                      + ( static_cast<scalar>( db ) + cell_atoms[i][1] ) * bravais_vectors[1]
                                      + ( static_cast<scalar>( dc ) + cell_atoms[i][2] ) * bravais_vectors[2] );

                            std::string message = fmt::format(
                                "Unable to initialize spin-system, because for a translation vector ({} {} {}), spins "
                                "{} and {} of the basis-cell occupy the same absolute position ({}) within a margin of "
                                "{} Angstrom. Please check the config file!",
                                da, db, dc, i, j, position.transpose(), epsilon );

                            spirit_throw(
                                Utility::Exception_Classifier::System_not_Initialized, Utility::Log_Level::Severe,
                                message );
                        }
                    }
                }
            }
        }
    }

    // Generate positions
    for( std::int64_t dc = 0; dc < n_cells[2]; ++dc )
    {
        for( std::int64_t db = 0; db < n_cells[1]; ++db )
        {
            for( std::int64_t da = 0; da < n_cells[0]; ++da )
            {
                for( std::int64_t iatom = 0; iatom < n_cell_atoms; ++iatom )
                {
                    std::int64_t ispin = iatom + dc * n_cell_atoms * n_cells[1] * n_cells[0]
                                         + db * n_cell_atoms * n_cells[0] + da * n_cell_atoms;

                    positions[ispin]
                        = lattice_constant
                          * ( ( static_cast<scalar>( da ) + cell_atoms[iatom][0] ) * bravais_vectors[0]
                              + ( static_cast<scalar>( db ) + cell_atoms[iatom][1] ) * bravais_vectors[1]
                              + ( static_cast<scalar>( dc ) + cell_atoms[iatom][2] ) * bravais_vectors[2] );
                }
            }
        }
    }
}

std::vector<tetrahedron_t> compute_delaunay_triangulation_3D( const std::vector<vector3_t> & points )
try
{
    const int ndim = 3;
    std::vector<tetrahedron_t> tetrahedra;
    tetrahedron_t tmp_tetrahedron;
    int * current_index = nullptr;

    orgQhull::Qhull qhull;
    qhull.runQhull( "", ndim, points.size(), (coordT *)points.data(), "d Qt Qbb Qz" );
    orgQhull::QhullFacetList facet_list = qhull.facetList();
    for( const auto & facet : facet_list )
    {
        if( !facet.isUpperDelaunay() )
        {
            current_index = &tmp_tetrahedron[0];
            for( const auto & vertex : facet.vertices() )
            {
                *current_index++ = vertex.point().id();
            }
            tetrahedra.push_back( tmp_tetrahedron );
        }
    }
    return tetrahedra;
}
catch( ... )
{
    spirit_handle_exception_core(
        "Could not compute 3D Delaunay triangulation of the Geometry. Probably Qhull threw an exception." );
    return std::vector<tetrahedron_t>( 0 );
}

std::vector<triangle_t> compute_delaunay_triangulation_2D( const std::vector<vector2_t> & points )
try
{
    const int ndim = 2;
    std::vector<triangle_t> triangles;
    triangle_t tmp_triangle;
    int * current_index = nullptr;

    orgQhull::Qhull qhull;
    qhull.runQhull( "", ndim, points.size(), (coordT *)points.data(), "d Qt Qbb Qz" );
    for( const auto & facet : qhull.facetList() )
    {
        if( !facet.isUpperDelaunay() )
        {
            current_index = &tmp_triangle[0];
            for( const auto & vertex : facet.vertices() )
            {
                *current_index++ = vertex.point().id();
            }
            triangles.push_back( tmp_triangle );
        }
    }
    return triangles;
}
catch( ... )
{
    spirit_handle_exception_core(
        "Could not compute 2D Delaunay triangulation of the Geometry. Probably Qhull threw an exception." );
    return std::vector<triangle_t>( 0 );
}

const std::vector<triangle_t> & Geometry::triangulation( int n_cell_step, std::array<int, 6> ranges )
{
    // Only every n_cell_step'th cell is used. So we check if there is still enough cells in all
    //      directions. Note: when visualising, 'n_cell_step' can be used to e.g. olny visualise
    //      every 2nd spin.
    if( ( n_cells[0] / n_cell_step < 2 && n_cells[0] > 1 ) || ( n_cells[1] / n_cell_step < 2 && n_cells[1] > 1 )
        || ( n_cells[2] / n_cell_step < 2 && n_cells[2] > 1 ) )
    {
        _triangulation.clear();
        return _triangulation;
    }

    // 2D: triangulation
    if( this->dimensionality == 2 )
    {
        // Check if the tetrahedra for this combination of n_cells and n_cell_step has already been calculated
        if( this->last_update_n_cell_step != n_cell_step || this->last_update_n_cells[0] != n_cells[0]
            || this->last_update_n_cells[1] != n_cells[1] || this->last_update_n_cells[2] != n_cells[2]
            || this->last_update_cell_ranges != ranges )
        {
            this->last_update_n_cell_step = n_cell_step;
            this->last_update_n_cells[0]  = n_cells[0];
            this->last_update_n_cells[1]  = n_cells[1];
            this->last_update_n_cells[2]  = n_cells[2];
            this->last_update_cell_ranges = ranges;

            _triangulation.clear();

            std::vector<vector2_t> points;

            std::int64_t a_min = ( ranges[0] >= 0 ) && ( ranges[0] <= n_cells[0] ) ? ranges[0] : 0;
            std::int64_t a_max = ( ranges[1] >= 0 ) && ( ranges[1] <= n_cells[0] ) ? ranges[1] : n_cells[0];
            std::int64_t b_min = ( ranges[2] >= 0 ) && ( ranges[2] <= n_cells[1] ) ? ranges[2] : 0;
            std::int64_t b_max = ( ranges[3] >= 0 ) && ( ranges[3] <= n_cells[1] ) ? ranges[3] : n_cells[1];
            std::int64_t c_min = ( ranges[4] >= 0 ) && ( ranges[4] <= n_cells[2] ) ? ranges[4] : 0;
            std::int64_t c_max = ( ranges[5] >= 0 ) && ( ranges[5] <= n_cells[2] ) ? ranges[5] : n_cells[2];

            std::int64_t n_a = std::max(
                static_cast<std::int64_t>( 1 ),
                static_cast<std::int64_t>(
                    std::ceil( static_cast<double>( ( a_max - a_min ) ) / static_cast<double>( n_cell_step ) ) ) );
            std::int64_t n_b = std::max(
                static_cast<std::int64_t>( 1 ),
                static_cast<std::int64_t>(
                    std::ceil( static_cast<double>( ( b_max - b_min ) ) / static_cast<double>( n_cell_step ) ) ) );
            std::int64_t n_c = std::max(
                static_cast<std::int64_t>( 1 ),
                static_cast<std::int64_t>(
                    std::ceil( static_cast<double>( ( c_max - c_min ) ) / static_cast<double>( n_cell_step ) ) ) );

            std::int64_t n_points = n_cell_atoms * n_a * n_b * n_c;

            if( ( ( n_a <= 1 || n_b <= 1 ) && dimensionality_basis != 2 ) || n_points < 3 )
            {
                _triangulation.clear();
                return _triangulation;
            }
            points.resize( n_points );

            // TODO: it seems the following is effectively just vec<vector3_t> (double) = vec<Vector3> (scalar)
            std::int64_t icell = 0, idx = 0;
            for( std::int64_t cell_c = c_min; cell_c < c_max; cell_c += n_cell_step )
            {
                for( std::int64_t cell_b = b_min; cell_b < b_max; cell_b += n_cell_step )
                {
                    for( std::int64_t cell_a = a_min; cell_a < a_max; cell_a += n_cell_step )
                    {
                        for( std::int64_t ibasis = 0; ibasis < n_cell_atoms; ++ibasis )
                        {
                            idx = ibasis + n_cell_atoms * cell_a + n_cell_atoms * n_cells[0] * cell_b
                                  + n_cell_atoms * n_cells[0] * n_cells[1] * cell_c;
                            points[icell].x = static_cast<double>( positions[idx][0] );
                            points[icell].y = static_cast<double>( positions[idx][1] );
                            ++icell;
                        }
                    }
                }
            }
            _triangulation = compute_delaunay_triangulation_2D( points );
        }
    } // endif 2D
    // 0D, 1D and 3D give no triangulation
    else
    {
        _triangulation.clear();
    }
    return _triangulation;
}

const std::vector<tetrahedron_t> & Geometry::tetrahedra( int n_cell_step, std::array<int, 6> ranges )
{
    // Only every n_cell_step'th cell is used. So we check if there is still enough cells in all directions. Note: when
    // visualising, 'n_cell_step' can be used to e.g. olny visualise every 2nd spin.
    if( ( n_cells[0] / n_cell_step < 2 ) || ( n_cells[1] / n_cell_step < 2 ) || ( n_cells[2] / n_cell_step < 2 ) )
    {
        _tetrahedra.clear();
        return _tetrahedra;
    }

    // --- 0-2 D gives no tetrahedra
    if( this->dimensionality != 3 )
    {
        _tetrahedra.clear();
        return _tetrahedra;
    }

    // --- 3D: Tetrahedra
    // Check if the tetrahedra for this combination of n_cells and n_cell_step has already been calculated
    if( ( this->last_update_n_cell_step != n_cell_step ) || ( this->last_update_n_cells[0] != n_cells[0] )
        || ( this->last_update_n_cells[1] != n_cells[1] ) || ( this->last_update_n_cells[2] != n_cells[2] )
        || ( this->last_update_cell_ranges != ranges ) )
    {
        this->last_update_n_cell_step = n_cell_step;
        this->last_update_n_cells[0]  = n_cells[0];
        this->last_update_n_cells[1]  = n_cells[1];
        this->last_update_n_cells[2]  = n_cells[2];
        this->last_update_cell_ranges = ranges;

        // Calculate the number of lattice translations covered by the given ranges
        std::int64_t a_min = ( ranges[0] >= 0 ) && ( ranges[0] <= n_cells[0] ) ? ranges[0] : 0;
        std::int64_t a_max = ( ranges[1] >= 0 ) && ( ranges[1] <= n_cells[0] ) ? ranges[1] : n_cells[0];
        std::int64_t b_min = ( ranges[2] >= 0 ) && ( ranges[2] <= n_cells[1] ) ? ranges[2] : 0;
        std::int64_t b_max = ( ranges[3] >= 0 ) && ( ranges[3] <= n_cells[1] ) ? ranges[3] : n_cells[1];
        std::int64_t c_min = ( ranges[4] >= 0 ) && ( ranges[4] <= n_cells[2] ) ? ranges[4] : 0;
        std::int64_t c_max = ( ranges[5] >= 0 ) && ( ranges[5] <= n_cells[2] ) ? ranges[5] : n_cells[2];

        std::int64_t n_a = std::max(
            static_cast<std::int64_t>( 1 ),
            static_cast<std::int64_t>(
                std::ceil( static_cast<double>( ( a_max - a_min ) ) / static_cast<double>( n_cell_step ) ) ) );
        std::int64_t n_b = std::max(
            static_cast<std::int64_t>( 1 ),
            static_cast<std::int64_t>(
                std::ceil( static_cast<double>( ( b_max - b_min ) ) / static_cast<double>( n_cell_step ) ) ) );
        std::int64_t n_c = std::max(
            static_cast<std::int64_t>( 1 ),
            static_cast<std::int64_t>(
                std::ceil( static_cast<double>( ( c_max - c_min ) ) / static_cast<double>( n_cell_step ) ) ) );
        std::int64_t n_points = n_cell_atoms * n_a * n_b * n_c;

        // The system can only be planar if n_points < 4 or if the basis cell is planar or linear or a single point and
        // at least one of the translation directions is not used
        if( ( ( n_a <= 1 || n_b <= 1 || n_c <= 1 ) && dimensionality_basis < 3 ) || n_points < 4 )
        {
            _tetrahedra.clear();
            return _tetrahedra;
        }

        // If we have only one spin in the basis our lattice is a simple regular geometry meaning everything can be
        // calculated by hand
        if( n_cell_atoms == 1 )
        {
            _tetrahedra.clear();
            std::array<int, 24> cell_indices{ 0, 1, 5, 3, 1, 3, 2, 5, 3, 2, 5, 6, 7, 6, 5, 3, 4, 7, 5, 3, 0, 4, 3, 5 };
            int x_offset = 1;
            int y_offset = n_a;
            int z_offset = n_a * n_b;
            std::array<int, 8> offsets{
                0,
                x_offset,
                x_offset + y_offset,
                y_offset,
                z_offset,
                x_offset + z_offset,
                x_offset + y_offset + z_offset,
                y_offset + z_offset,
            };

            for( int ix = 0; ix < n_a - 1; ix++ )
            {
                for( int iy = 0; iy < n_b - 1; iy++ )
                {
                    for( int iz = 0; iz < n_c - 1; iz++ )
                    {
                        int base_index = ix * x_offset + iy * y_offset + iz * z_offset;
                        for( std::uint8_t j = 0; j < 6; j++ )
                        {
                            tetrahedron_t tetrahedron;
                            for( std::uint8_t k = 0; k < 4; k++ )
                            {
                                tetrahedron[k] = base_index + offsets[cell_indices[j * 4 + k]];
                            }
                            _tetrahedra.push_back( tetrahedron );
                        }
                    }
                }
            }
        }
        // For general basis cells we calculate the Delaunay tetrahedra
        else
        {
            // TODO: it seems the following is effectively just vec<vector3_t> (double) = vec<Vector3> (scalar)
            std::vector<vector3_t> points( n_points );
            std::int64_t icell = 0, idx = 0;
            for( std::int64_t cell_c = c_min; cell_c < c_max; cell_c += n_cell_step )
            {
                for( std::int64_t cell_b = b_min; cell_b < b_max; cell_b += n_cell_step )
                {
                    for( std::int64_t cell_a = a_min; cell_a < a_max; cell_a += n_cell_step )
                    {
                        for( std::int64_t ibasis = 0; ibasis < n_cell_atoms; ++ibasis )
                        {
                            idx = ibasis + n_cell_atoms * cell_a + n_cell_atoms * n_cells[0] * cell_b
                                  + n_cell_atoms * n_cells[0] * n_cells[1] * cell_c;
                            points[icell].x = static_cast<double>( positions[idx][0] );
                            points[icell].y = static_cast<double>( positions[idx][1] );
                            points[icell].z = static_cast<double>( positions[idx][2] );
                            ++icell;
                        }
                    }
                }
            }
            _tetrahedra = compute_delaunay_triangulation_3D( points );
        }
    }

    return _tetrahedra;
}

std::vector<Vector3> Geometry::BravaisVectorsSC()
{
    return {
        { scalar( 1 ), scalar( 0 ), scalar( 0 ) },
        { scalar( 0 ), scalar( 1 ), scalar( 0 ) },
        { scalar( 0 ), scalar( 0 ), scalar( 1 ) },
    };
}

std::vector<Vector3> Geometry::BravaisVectorsFCC()
{
    return {
        { scalar( 0.5 ), scalar( 0.0 ), scalar( 0.5 ) },
        { scalar( 0.5 ), scalar( 0.5 ), scalar( 0.0 ) },
        { scalar( 0.0 ), scalar( 0.5 ), scalar( 0.5 ) },
    };
}

std::vector<Vector3> Geometry::BravaisVectorsBCC()
{
    return {
        { scalar( 0.5 ), scalar( 0.5 ), scalar( -0.5 ) },
        { scalar( -0.5 ), scalar( 0.5 ), scalar( -0.5 ) },
        { scalar( 0.5 ), scalar( -0.5 ), scalar( -0.5 ) },
    };
}

std::vector<Vector3> Geometry::BravaisVectorsHex2D60()
{
    return {
        { scalar( 0.5 * std::sqrt( 3 ) ), scalar( -0.5 ), scalar( 0 ) },
        { scalar( 0.5 * std::sqrt( 3 ) ), scalar( 0.5 ), scalar( 0 ) },
        { scalar( 0 ), scalar( 0 ), scalar( 1 ) },
    };
}

std::vector<Vector3> Geometry::BravaisVectorsHex2D120()
{
    return {
        { scalar( 0.5 ), scalar( -0.5 * std::sqrt( 3 ) ), scalar( 0 ) },
        { scalar( 0.5 ), scalar( 0.5 * std::sqrt( 3 ) ), scalar( 0 ) },
        { scalar( 0 ), scalar( 0 ), scalar( 1 ) },
    };
}

void Geometry::applyCellComposition()
{
    std::int64_t N     = this->n_cell_atoms;
    std::int64_t Na    = this->n_cells[0];
    std::int64_t Nb    = this->n_cells[1];
    std::int64_t Nc    = this->n_cells[2];
    std::int64_t ispin = 0, iatom = 0;
    scalar concentration = 0, rvalue = 0;
    std::vector<bool> visited( N );

    std::mt19937 prng;
    std::uniform_real_distribution<scalar> distribution;
    if( this->cell_composition.disordered )
    {
        // TODO: the seed should be a parameter and the instance a member of this class
        prng         = std::mt19937( 2006 );
        distribution = std::uniform_real_distribution<scalar>( 0, 1 );
        // In the disordered case, unvisited atoms will be vacancies
        this->atom_types = intfield( nos, -1 );
    }

    for( std::int64_t na = 0; na < Na; ++na )
    {
        for( std::int64_t nb = 0; nb < Nb; ++nb )
        {
            for( std::int64_t nc = 0; nc < Nc; ++nc )
            {
                std::fill( visited.begin(), visited.end(), false );

                for( std::size_t icomposition = 0; icomposition < this->cell_composition.iatom.size(); ++icomposition )
                {
                    iatom = this->cell_composition.iatom[icomposition];

                    if( !visited[iatom] )
                    {
                        ispin = N * na + N * Na * nb + N * Na * Nb * nc + iatom;

                        // In the disordered case, we only visit an atom if the dice will it
                        if( this->cell_composition.disordered )
                        {
                            concentration = this->cell_composition.concentration[icomposition];
                            rvalue        = distribution( prng );
                            if( rvalue <= concentration )
                            {
                                this->atom_types[ispin] = this->cell_composition.atom_type[icomposition];
                                this->mu_s[ispin]       = this->cell_composition.mu_s[icomposition];
                                visited[iatom]          = true;
                                if( this->atom_types[ispin] < 0 )
                                    --this->nos_nonvacant;
                            }
                        }
                        // In the ordered case, we visit every atom
                        else
                        {
                            this->atom_types[ispin] = this->cell_composition.atom_type[icomposition];
                            this->mu_s[ispin]       = this->cell_composition.mu_s[icomposition];
                            visited[iatom]          = true;
                            if( this->atom_types[ispin] < 0 )
                                --this->nos_nonvacant;
                        }

                        // Pinning of boundary layers
                        if( ( na < pinning.na_left || na >= Na - pinning.na_right )
                            || ( nb < pinning.nb_left || nb >= Nb - pinning.nb_right )
                            || ( nc < pinning.nc_left || nc >= Nc - pinning.nc_right ) )
                        {
                            // Pinned cells
                            this->mask_unpinned[ispin]     = 0;
                            this->mask_pinned_cells[ispin] = pinning.pinned_cell[iatom];
                        }
                    }
                }
            }
        }
    }
}

void Geometry::calculateDimensionality()
{
    Vector3 test_vec_basis, test_vec_translations;

    const scalar epsilon = std::numeric_limits<scalar>::epsilon();

    // ----- Find dimensionality of the basis -----
    if( n_cell_atoms == 1 )
    {
        dimensionality_basis = 0;
    }
    else if( n_cell_atoms == 2 )
    {
        dimensionality_basis = 1;
        test_vec_basis       = positions[0] - positions[1];
    }
    else
    {
        // Get basis atoms relative to the first atom
        Vector3 v0 = positions[0];
        std::vector<Vector3> b_vectors( n_cell_atoms - 1 );
        for( std::int64_t i = 1; i < n_cell_atoms; ++i )
            b_vectors[i - 1] = ( positions[i] - v0 ).normalized();

        // Calculate basis dimensionality
        // test vec is along line
        test_vec_basis = b_vectors[0];
        //      is it 1D?
        std::size_t n_parallel = 0;
        for( std::size_t i = 1; i < b_vectors.size(); ++i )
        {
            if( 1 - std::abs( b_vectors[i].dot( test_vec_basis ) ) < epsilon )
                ++n_parallel;
            // Else n_parallel will give us the last parallel vector
            // Also the if-statement for dimensionality_basis=1 wont be met
            else
                break;
        }
        if( n_parallel == b_vectors.size() - 1 )
        {
            dimensionality_basis = 1;
        }
        else
        {
            // test vec is normal to plane
            test_vec_basis = b_vectors[0].cross( b_vectors[n_parallel + 1] );
            //      is it 2D?
            std::size_t n_in_plane = 0;
            for( std::size_t i = 2; i < b_vectors.size(); ++i )
            {
                if( std::abs( b_vectors[i].dot( test_vec_basis ) ) < epsilon )
                    ++n_in_plane;
            }
            if( static_cast<std::int64_t>( n_in_plane ) == static_cast<std::int64_t>( b_vectors.size() ) - 2 )
                dimensionality_basis = 2;
            else
            {
                this->dimensionality_basis = 3;
                this->dimensionality       = 3;
                return;
            }
        }
    }

    // ----- Find dimensionality of the translations -----
    // The following are zero if the corresponding pair is parallel or antiparallel
    double t01 = std::abs( bravais_vectors[0].normalized().dot( bravais_vectors[1].normalized() ) ) - 1.0;
    double t02 = std::abs( bravais_vectors[0].normalized().dot( bravais_vectors[2].normalized() ) ) - 1.0;
    double t12 = std::abs( bravais_vectors[1].normalized().dot( bravais_vectors[2].normalized() ) ) - 1.0;
    // Check if pairs are linearly independent
    std::uint8_t dims_translations   = 0;
    std::uint8_t n_independent_pairs = 0;
    if( ( t01 < epsilon ) && ( n_cells[0] > 1 ) && ( n_cells[1] > 1 ) )
        ++n_independent_pairs;
    if( ( t02 < epsilon ) && ( n_cells[0] > 1 ) && ( n_cells[2] > 1 ) )
        ++n_independent_pairs;
    if( ( t12 < epsilon ) && ( n_cells[1] > 1 ) && ( n_cells[2] > 1 ) )
        ++n_independent_pairs;
    // Calculate translations dimensionality
    if( ( n_cells[0] == 1 ) && ( n_cells[1] == 1 ) && ( n_cells[2] == 1 ) )
    {
        dims_translations = 0;
    }
    else if( n_independent_pairs == 0 )
    {
        dims_translations = 1;
        // Test if vec is along the line
        for( std::uint8_t i = 0; i < 3; ++i )
            if( n_cells[i] > 1 )
                test_vec_translations = bravais_vectors[i];
    }
    else if( n_independent_pairs < 3 )
    {
        dims_translations = 2;
        // Test if vec is normal to plane
        std::uint8_t n = 0;
        std::vector<Vector3> plane( 2 );
        for( std::uint8_t i = 0; i < 3; ++i )
        {
            if( n_cells[i] > 1 )
            {
                plane[n] = bravais_vectors[i];
                ++n;
            }
        }
        test_vec_translations = plane[0].cross( plane[1] );
    }
    else
    {
        this->dimensionality = 3;
        return;
    }

    // ----- Calculate dimensionality of system -----
    test_vec_basis.normalize();
    test_vec_translations.normalize();
    // If one dimensionality is zero, only the other counts
    if( dimensionality_basis == 0 )
    {
        this->dimensionality = dims_translations;
        return;
    }
    else if( dims_translations == 0 )
    {
        this->dimensionality = dimensionality_basis;
        return;
    }
    // If both are linear or both are planar, the test vectors should be (anti)parallel if the geometry is 1D or 2D
    else if( dimensionality_basis == dims_translations )
    {
        // If they are parallel the system has the dimensionality of the basis
        if( std::abs( test_vec_basis.dot( test_vec_translations ) ) - 1 < epsilon )
        {
            this->dimensionality = dimensionality_basis;
            return;
        }
        // If not, the dimensionality is increased by 1
        else
        {
            this->dimensionality = dimensionality_basis + 1;
            return;
        }
    }
    // If one is linear (1D), and the other planar (2D) then the test vectors should be orthogonal if the geometry is 2D
    else if(
        ( dimensionality_basis == 1 && dims_translations == 2 )
        || ( dimensionality_basis == 2 && dims_translations == 1 ) )
    {
        if( std::abs( test_vec_basis.dot( test_vec_translations ) ) < epsilon )
        {
            this->dimensionality = 2;
            return;
        }
        else
        {
            this->dimensionality = 3;
            return;
        }
    }
}

void Geometry::calculateBounds()
{
    this->bounds_max.setZero();
    this->bounds_min.setZero();
    for( std::size_t iatom = 0; iatom < static_cast<std::size_t>( nos ); ++iatom )
    {
        for( std::uint8_t dim = 0; dim < 3; ++dim )
        {
            if( this->positions[iatom][dim] < this->bounds_min[dim] )
                this->bounds_min[dim] = this->positions[iatom][dim];
            if( this->positions[iatom][dim] > this->bounds_max[dim] )
                this->bounds_max[dim] = this->positions[iatom][dim];
        }
    }
}

void Geometry::calculateUnitCellBounds()
{
    this->cell_bounds_max.setZero();
    this->cell_bounds_min.setZero();
    for( const auto & bravais_vector : this->bravais_vectors )
    {
        for( std::int64_t iatom = 0; iatom < this->n_cell_atoms; ++iatom )
        {
            Vector3 neighbour1 = this->positions[iatom] + this->lattice_constant * bravais_vector;
            Vector3 neighbour2 = this->positions[iatom] - this->lattice_constant * bravais_vector;
            for( std::uint8_t dim = 0; dim < 3; ++dim )
            {
                if( neighbour1[dim] < this->cell_bounds_min[dim] )
                    this->cell_bounds_min[dim] = neighbour1[dim];
                if( neighbour1[dim] > this->cell_bounds_max[dim] )
                    this->cell_bounds_max[dim] = neighbour1[dim];
                if( neighbour2[dim] < this->cell_bounds_min[dim] )
                    this->cell_bounds_min[dim] = neighbour2[dim];
                if( neighbour2[dim] > this->cell_bounds_max[dim] )
                    this->cell_bounds_max[dim] = neighbour2[dim];
            }
        }
    }
    this->cell_bounds_min *= 0.5;
    this->cell_bounds_max *= 0.5;
}

void Geometry::calculateGeometryType()
{
    const scalar epsilon = std::numeric_limits<scalar>::epsilon();

    // Automatically try to determine GeometryType
    // Single-atom unit cell
    if( cell_atoms.size() == 1 )
    {
        // If the basis vectors are orthogonal, it is a rectilinear lattice
        if( ( std::abs( bravais_vectors[0].normalized().dot( bravais_vectors[1].normalized() ) ) < epsilon )
            && ( std::abs( bravais_vectors[0].normalized().dot( bravais_vectors[2].normalized() ) ) < epsilon ) )
        {
            // If equidistant it is simple cubic
            if( ( bravais_vectors[0].norm() == bravais_vectors[1].norm() )
                && ( bravais_vectors[1].norm() == bravais_vectors[2].norm() ) )
                this->classifier = BravaisLatticeType::SC;
            // Otherwise only rectilinear
            else
                this->classifier = BravaisLatticeType::Rectilinear;
        }
    }
    // Regular unit cell with multiple atoms (e.g. bcc, fcc, hex)
    // else if (n_cell_atoms == 2)
    // Irregular unit cells arranged on a lattice (e.g. B20 or custom)
    /*else if (n_cells[0] > 1 || n_cells[1] > 1 || n_cells[2] > 1)
    {
        this->classifier = BravaisLatticeType::Lattice;
    }*/
    // A single irregular unit cell
    else
    {
        this->classifier = BravaisLatticeType::Irregular;
    }
}

void Geometry::Apply_Pinning( vectorfield & vf )
{
#if defined( SPIRIT_ENABLE_PINNING )
    std::int64_t N     = this->n_cell_atoms;
    std::int64_t Na    = this->n_cells[0];
    std::int64_t Nb    = this->n_cells[1];
    std::int64_t Nc    = this->n_cells[2];
    std::int64_t ispin = 0;

    for( std::int64_t iatom = 0; iatom < N; ++iatom )
    {
        for( std::int64_t na = 0; na < Na; ++na )
        {
            for( std::int64_t nb = 0; nb < Nb; ++nb )
            {
                for( std::int64_t nc = 0; nc < Nc; ++nc )
                {
                    ispin = N * na + N * Na * nb + N * Na * Nb * nc + iatom;
                    if( this->mask_unpinned[ispin] == 0 )
                        vf[ispin] = this->mask_pinned_cells[ispin];
                }
            }
        }
    }
#endif
}

} // namespace Data