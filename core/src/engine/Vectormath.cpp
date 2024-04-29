#include <engine/Indexing.hpp>
#include <engine/Manifoldmath.hpp>
#include <engine/Vectormath.hpp>
#include <utility/Constants.hpp>
#include <utility/Exception.hpp>
#include <utility/Logging.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <array>

using namespace Utility;
using Utility::Constants::Pi;

#ifndef SPIRIT_USE_CUDA

namespace Engine
{
namespace Vectormath
{

void get_random_vector( std::uniform_real_distribution<scalar> & distribution, std::mt19937 & prng, Vector3 & vec )
{
    for( int dim = 0; dim < 3; ++dim )
    {
        vec[dim] = distribution( prng );
    }
}
void get_random_vectorfield( std::mt19937 & prng, vectorfield & xi )
{
    // PRNG gives RN [-1,1] -> multiply with epsilon
    auto distribution = std::uniform_real_distribution<scalar>( -1, 1 );
// TODO: parallelization of this is actually not quite so trivial
#pragma omp parallel for
    for( unsigned int i = 0; i < xi.size(); ++i )
    {
        get_random_vector( distribution, prng, xi[i] );
    }
}

void get_random_vector_unitsphere(
    std::uniform_real_distribution<scalar> & distribution, std::mt19937 & prng, Vector3 & vec )
{
    scalar v_z = distribution( prng );
    scalar phi = distribution( prng ) * Pi;

    scalar r_xy = std::sqrt( 1 - v_z * v_z );

    vec[0] = r_xy * std::cos( phi );
    vec[1] = r_xy * std::sin( phi );
    vec[2] = v_z;
}
void get_random_vectorfield_unitsphere( std::mt19937 & prng, vectorfield & xi )
{
    // PRNG gives RN [-1,1] -> multiply with epsilon
    auto distribution = std::uniform_real_distribution<scalar>( -1, 1 );
// TODO: parallelization of this is actually not quite so trivial
#pragma omp parallel for
    for( unsigned int i = 0; i < xi.size(); ++i )
    {
        get_random_vector_unitsphere( distribution, prng, xi[i] );
    }
}

} // namespace Vectormath

} // namespace Engine

#endif

namespace Engine
{

namespace Vectormath
{

// out[i] += c*a
void add_c_a( const scalar & c, const Vector3 & vec, vectorfield & out )
{
    const Vector3 a = c * vec;
    Backend::for_each( SPIRIT_PAR out.begin(), out.end(), [a] SPIRIT_LAMBDA( Vector3 & v ) { v += a; } );
}

// out[i] += c*a[i]
void add_c_a( const scalar & c, const vectorfield & vf, vectorfield & out )
{
    Backend::transform(
        SPIRIT_PAR out.begin(), out.end(), vf.begin(), out.begin(),
        [c] SPIRIT_LAMBDA( const Vector3 & res, const Vector3 & a ) -> Vector3 { return res + c * a; } );
}

void add_c_a( const scalar & c, const vectorfield & vf, vectorfield & out, const intfield & mask )
{
    const auto * v = raw_pointer_cast( vf.data() );
    const auto * m = raw_pointer_cast( mask.data() );
    auto * o       = raw_pointer_cast( out.data() );
    Backend::for_each_n(
        Backend::make_counting_iterator( 0 ), out.size(),
        [c, v, m, o] SPIRIT_LAMBDA( const int idx ) { o[idx] += m[idx] * c * v[idx]; } );
}

// out[i] += c[i]*a[i]
void add_c_a( const scalarfield & c, const vectorfield & vf, vectorfield & out )
{
    const auto * cc = raw_pointer_cast( c.data() );
    const auto * v  = raw_pointer_cast( vf.data() );
    auto * o        = raw_pointer_cast( out.data() );
    Backend::for_each_n(
        Backend::make_counting_iterator( 0 ), out.size(),
        [v, cc, o] SPIRIT_LAMBDA( const int idx ) { o[idx] += cc[idx] * v[idx]; } );
}

// out[i] = c*a
void set_c_a( const scalar & c, const Vector3 & vec, vectorfield & out )
{
    Vectormath::fill( out, Vector3( c * vec ) );
}

// out[i] = c*a
void set_c_a( const scalar & c, const Vector3 & vec, vectorfield & out, const intfield & mask )
{
    Vectormath::fill( out, Vector3( c * vec ), mask );
}

// out[i] = c*a[i]
void set_c_a( const scalar & c, const vectorfield & vf, vectorfield & out )
{
    Backend::transform( SPIRIT_PAR vf.begin(), vf.end(), out.begin(), Backend::scale( c ) );
}

// out[i] = c*a[i]
void set_c_a( const scalar & c, const vectorfield & vf, vectorfield & out, const intfield & mask )
{
    Backend::transform(
        SPIRIT_PAR vf.begin(), vf.end(), mask.begin(), out.begin(),
        [c] SPIRIT_LAMBDA( const Vector3 & a, const int cond ) -> Vector3 { return c * cond * a; } );
}
// out[i] = c[i]*a[i]
void set_c_a( const scalarfield & c, const vectorfield & vf, vectorfield & out )
{
    Backend::transform(
        SPIRIT_PAR vf.begin(), vf.end(), c.begin(), out.begin(),
        [] SPIRIT_LAMBDA( const Vector3 & a, const scalar c ) -> Vector3 { return c * a; } );
}

// out[i] += c * a*b[i]
void add_c_dot( const scalar & c, const Vector3 & vec, const vectorfield & vf, scalarfield & out )
{
    Backend::transform(
        SPIRIT_PAR vf.begin(), vf.end(), out.begin(), out.begin(),
        [c, vec] SPIRIT_LAMBDA( const Vector3 & a, const scalar res ) -> scalar { return res + c * vec.dot( a ); } );
}
// out[i] += c * a[i]*b[i]
void add_c_dot( const scalar & c, const vectorfield & vf1, const vectorfield & vf2, scalarfield & out )
{
    const auto * v1 = raw_pointer_cast( vf1.data() );
    const auto * v2 = raw_pointer_cast( vf2.data() );
    auto * o        = raw_pointer_cast( out.data() );
    Backend::for_each_n(
        SPIRIT_PAR Backend::make_counting_iterator( 0 ), out.size(),
        [c, v1, v2, o] SPIRIT_LAMBDA( const int idx ) { o[idx] += c * v1[idx].dot( v2[idx] ); } );
}

// out[i] = c * a*b[i]
void set_c_dot( const scalar & c, const Vector3 & a, const vectorfield & b, scalarfield & out )
{
    Backend::transform(
        SPIRIT_PAR b.begin(), b.end(), out.begin(),
        [c, a] SPIRIT_LAMBDA( const Vector3 & b ) -> scalar { return c * a.dot( b ); } );
}
// out[i] = c * a[i]*b[i]
void set_c_dot( const scalar & c, const vectorfield & a, const vectorfield & b, scalarfield & out )
{
    Backend::transform(
        SPIRIT_PAR a.begin(), a.end(), b.begin(), out.begin(),
        [c] SPIRIT_LAMBDA( const Vector3 & a, const Vector3 & b ) -> scalar { return c * a.dot( b ); } );
}

// out[i] += c * a x b[i]
void add_c_cross( const scalar & c, const Vector3 & a, const vectorfield & b, vectorfield & out )
{
    Backend::transform(
        SPIRIT_PAR b.begin(), b.end(), out.begin(), out.begin(),
        [c, a] SPIRIT_LAMBDA( const Vector3 & b, const Vector3 & res ) -> Vector3 { return res + c * a.cross( b ); } );
}
// out[i] += c * a[i] x b[i]
void add_c_cross( const scalar & c, const vectorfield & a, const vectorfield & b, vectorfield & out )
{
    const auto * aa = raw_pointer_cast( a.data() );
    const auto * bb = raw_pointer_cast( b.data() );
    auto * o        = raw_pointer_cast( out.data() );
    Backend::for_each_n(
        SPIRIT_PAR Backend::make_counting_iterator( 0 ), out.size(),
        [c, aa, bb, o] SPIRIT_LAMBDA( const int idx ) { o[idx] += c * aa[idx].cross( bb[idx] ); } );
}

// out[i] += c[i] * a[i] x b[i]
void add_c_cross( const scalarfield & c, const vectorfield & a, const vectorfield & b, vectorfield & out )
{
    const auto * cc = raw_pointer_cast( c.data() );
    const auto * aa = raw_pointer_cast( a.data() );
    const auto * bb = raw_pointer_cast( b.data() );
    auto * o        = raw_pointer_cast( out.data() );

    Backend::for_each_n(
        SPIRIT_PAR Backend::make_counting_iterator( 0 ), out.size(),
        [cc, aa, bb, o] SPIRIT_LAMBDA( const int idx ) { o[idx] += cc[idx] * aa[idx].cross( bb[idx] ); } );
}


// out[i] = c * a x b[i]
void set_c_cross( const scalar & c, const Vector3 & a, const vectorfield & b, vectorfield & out )
{
    Backend::transform(
        SPIRIT_PAR b.begin(), b.end(), out.begin(),
        [c, a] SPIRIT_LAMBDA( const Vector3 & b ) -> Vector3 { return c * a.cross( b ); } );
}

// out[i] = c * a[i] x b[i]
void set_c_cross( const scalar & c, const vectorfield & a, const vectorfield & b, vectorfield & out )
{
    Backend::transform(
        SPIRIT_PAR a.begin(), a.end(), b.begin(), out.begin(),
        [c] SPIRIT_LAMBDA( const Vector3 & a, const Vector3 & b ) -> Vector3 { return c * a.cross( b ); } );
}

// Constructs a rotation matrix that rotates to a frame with "normal" as the z-axis
Matrix3 dreibein( const Vector3 & normal )
{
    const Vector3 ex = { 1, 0, 0 };
    const Vector3 ey = { 0, 1, 0 };

    Vector3 x_hat;
    if( std::abs( normal.dot( ex ) - 1 ) > 1e-1 ) // Make sure not to take the cross product with a parallel vector
    {
        x_hat = normal.cross( ex );
    }
    else
    {
        x_hat = normal.cross( ey );
    }

    Vector3 y_hat = normal.cross( x_hat );

    Matrix3 dreibein;
    dreibein.row( 0 ) = x_hat.normalized();
    dreibein.row( 1 ) = y_hat.normalized();
    dreibein.row( 2 ) = normal.normalized();
    return dreibein;
}

/////////////////////////////////////////////////////////////////
scalar solid_angle_1( const Vector3 & v1, const Vector3 & v2, const Vector3 & v3 )
{
    // Get sign
    scalar pm = v1.dot( v2.cross( v3 ) );
    if( pm != 0 )
        pm /= std::abs( pm );

    // angle
    scalar solid_angle = ( 1 + v1.dot( v2 ) + v2.dot( v3 ) + v3.dot( v1 ) )
                         / std::sqrt( 2 * ( 1 + v1.dot( v2 ) ) * ( 1 + v2.dot( v3 ) ) * ( 1 + v3.dot( v1 ) ) );
    if( solid_angle == 1 )
        solid_angle = 0;
    else
        solid_angle = pm * 2 * std::acos( solid_angle );

    return solid_angle;
}

scalar solid_angle_2( const Vector3 & v1, const Vector3 & v2, const Vector3 & v3 )
{
    // Using the solid angle formula by Oosterom and Strackee (note we assume vectors to be normalized to 1)
    // https://en.wikipedia.org/wiki/Solid_angle#Tetrahedron

    scalar x           = v1.dot( v2.cross( v3 ) );
    scalar y           = 1 + v1.dot( v2 ) + v1.dot( v3 ) + v2.dot( v3 );
    scalar solid_angle = 2 * std::atan2( x, y );

    return solid_angle;
}

void rotate( const Vector3 & v, const Vector3 & axis, const scalar & angle, Vector3 & v_out )
{
    v_out = v * std::cos( angle ) + axis.cross( v ) * std::sin( angle )
            + axis * axis.dot( v ) * ( 1 - std::cos( angle ) );
}

// XXX: should we add test for that function since it's calling the already tested rotat()
void rotate( const vectorfield & v, const vectorfield & axis, const scalarfield & angle, vectorfield & v_out )
{
    for( unsigned int i = 0; i < v_out.size(); i++ )
        rotate( v[i], axis[i], angle[i], v_out[i] );
}

Vector3 decompose( const Vector3 & v, const std::vector<Vector3> & basis )
{
    Eigen::Ref<const Matrix3> A = Eigen::Map<const Matrix3>( basis[0].data() );
    return A.colPivHouseholderQr().solve( v );
}

/////////////////////////////////////////////////////////////////

Vector3 Magnetization( const vectorfield & vf, const scalarfield & mu_s )
{
    return Backend::transform_reduce(
               SPIRIT_PAR vf.begin(), vf.end(), mu_s.begin(), zero_value<Vector3>(), Backend::plus<Vector3>{},
               [] SPIRIT_LAMBDA( const Vector3 & v, const scalar s ) { return s * v; } )
           / vf.size();
}

void TopologicalChargeDensity(
    const vectorfield & vf, const Data::Geometry & geometry, const intfield & boundary_conditions,
    scalarfield & charge_density, std::vector<int> & triangle_indices )
{
    charge_density.resize( 0 );

    // This implementations assumes
    // 1. No basis atom lies outside the cell spanned by the basis vectors of the lattice
    // 2. The geometry is a plane in x and y and spanned by the first 2 basis_vectors of the lattice
    // 3. The first basis atom lies at (0,0)

    const auto & positions = geometry.positions;
    scalar charge          = 0;

    // Compute Delaunay for unitcell + basis with neighbouring lattice sites in directions a, b, and a+b
    std::vector<Data::vector2_t> basis_cell_points( geometry.n_cell_atoms + 3 );
    for( int i = 0; i < geometry.n_cell_atoms; i++ )
    {
        basis_cell_points[i].x = double( positions[i][0] );
        basis_cell_points[i].y = double( positions[i][1] );
    }

    // To avoid cases where the basis atoms lie on the boundary of the convex hull the corners of the parallelogram
    // spanned by the lattice sites 0, a, b and a+b are stretched away from the center for the triangulation
    scalar stretch_factor = 0.1;

    // For the rare case where the first basis atoms does not lie at (0,0,0)
    Vector3 basis_offset = positions[0];

    Vector3 ta = geometry.lattice_constant * geometry.bravais_vectors[0];
    Vector3 tb = geometry.lattice_constant * geometry.bravais_vectors[1];
    Vector3 tc = geometry.lattice_constant * geometry.bravais_vectors[2];

    // basis_cell_points[0] coincides with the '0' lattice site (plus basis_offset)
    basis_cell_points[0].x -= stretch_factor * ( ta + tb )[0];
    basis_cell_points[0].y -= stretch_factor * ( ta + tb )[1];

    // a+b
    basis_cell_points[geometry.n_cell_atoms].x = double( ( ta + tb + positions[0] + stretch_factor * ( ta + tb ) )[0] );
    basis_cell_points[geometry.n_cell_atoms].y = double( ( ta + tb + positions[0] + stretch_factor * ( ta + tb ) )[1] );
    // b
    basis_cell_points[geometry.n_cell_atoms + 1].x = double( ( tb + positions[0] - stretch_factor * ( ta - tb ) )[0] );
    basis_cell_points[geometry.n_cell_atoms + 1].y = double( ( tb + positions[0] - stretch_factor * ( ta - tb ) )[1] );
    // a
    basis_cell_points[geometry.n_cell_atoms + 2].x = double( ( ta + positions[0] + stretch_factor * ( ta - tb ) )[0] );
    basis_cell_points[geometry.n_cell_atoms + 2].y = double( ( ta + positions[0] + stretch_factor * ( ta - tb ) )[1] );

    std::vector<Data::triangle_t> triangulation;
    triangulation = Data::compute_delaunay_triangulation_2D( basis_cell_points );

    for( Data::triangle_t tri : triangulation )
    {
        // Compute the sign of this triangle
        Vector3 triangle_normal;
        vectorfield tri_positions( 3 );
        for( int i = 0; i < 3; i++ )
            tri_positions[i]
                = { scalar( basis_cell_points[tri[i]].x ), scalar( basis_cell_points[tri[i]].y ), scalar( 0 ) };
        triangle_normal = ( tri_positions[0] - tri_positions[1] ).cross( tri_positions[0] - tri_positions[2] );
        triangle_normal.normalize();
        scalar sign = triangle_normal[2] / std::abs( triangle_normal[2] );

        // We try to apply the Delaunay triangulation at each bravais-lattice point
        // For each corner of the triangle we check wether it is "allowed" (which means either inside the simulation box
        // or permitted by periodic boundary conditions) Then we can add the top charge for all trios of spins connected
        // by this triangle
        for( int b = 0; b < geometry.n_cells[1]; ++b )
        {
            for( int a = 0; a < geometry.n_cells[0]; ++a )
            {
                std::array<Vector3, 3> tri_spins;
                std::array<int, 3> tri_indices;
                // bools to check wether it is allowed to take the next lattice site in direction a, b or a+b
                bool a_next_allowed = ( a + 1 < geometry.n_cells[0] || boundary_conditions[0] );
                bool b_next_allowed = ( b + 1 < geometry.n_cells[1] || boundary_conditions[1] );
                bool valid_triangle = true;
                for( int i = 0; i < 3; ++i )
                {
                    int idx;
                    if( tri[i] < geometry.n_cell_atoms ) // tri[i] is an index of a basis atom, no wrap around can occur
                    {
                        idx = ( tri[i] + a * geometry.n_cell_atoms + b * geometry.n_cell_atoms * geometry.n_cells[0] );
                    }
                    else if( tri[i] == geometry.n_cell_atoms + 2 && a_next_allowed ) // Translation by a
                    {
                        idx = ( ( a + 1 ) % geometry.n_cells[0] ) * geometry.n_cell_atoms
                              + b * geometry.n_cell_atoms * geometry.n_cells[0];
                    }
                    else if( tri[i] == geometry.n_cell_atoms + 1 && b_next_allowed ) // Translation by b
                    {
                        idx = a * geometry.n_cell_atoms
                              + ( ( b + 1 ) % geometry.n_cells[1] ) * geometry.n_cell_atoms * geometry.n_cells[0];
                    }
                    else if(
                        tri[i] == geometry.n_cell_atoms && a_next_allowed && b_next_allowed ) // Translation by a + b
                    {
                        idx = ( ( a + 1 ) % geometry.n_cells[0] ) * geometry.n_cell_atoms
                              + ( ( b + 1 ) % geometry.n_cells[1] ) * geometry.n_cell_atoms * geometry.n_cells[0];
                    }
                    else // Translation not allowed, skip to next triangle
                    {
                        valid_triangle = false;
                        break;
                    }
                    tri_spins[i]   = vf[idx];
                    tri_indices[i] = idx;
                }
                if( valid_triangle )
                {
                    triangle_indices.push_back( tri_indices[0] );
                    triangle_indices.push_back( tri_indices[1] );
                    triangle_indices.push_back( tri_indices[2] );
                    charge_density.push_back(
                        sign / ( 4.0 * Pi ) * solid_angle_2( tri_spins[0], tri_spins[1], tri_spins[2] ) );
                }
            }
        }
    }
}

// Calculate the topological charge inside a vectorfield
scalar TopologicalCharge( const vectorfield & vf, const Data::Geometry & geom, const intfield & boundary_conditions )
{
    scalarfield charge_density( 0 );
    std::vector<int> triangle_indices( 0 );
    TopologicalChargeDensity( vf, geom, boundary_conditions, charge_density, triangle_indices );
    return Vectormath::sum( charge_density );
}

void get_gradient_distribution(
    const Data::Geometry & geometry, Vector3 gradient_direction, scalar gradient_start, scalar gradient_inclination,
    scalarfield & distribution, scalar range_min, scalar range_max )
{
    // Ensure a normalized direction vector
    gradient_direction.normalize();

    // Basic linear gradient distribution
    set_c_dot( gradient_inclination, gradient_direction, geometry.positions, distribution );

    // Get the minimum (i.e. starting point) of the distribution
    scalar bmin     = geometry.bounds_min.dot( gradient_direction );
    scalar bmax     = geometry.bounds_max.dot( gradient_direction );
    scalar dist_min = std::min( bmin, bmax );
    // Set the starting point
    Vectormath::add( distribution, gradient_start - gradient_inclination * dist_min );

    // Cut off negative values
    set_range( distribution, range_min, range_max );
}

void directional_gradient(
    const vectorfield & vf, const Data::Geometry & geometry, const intfield & boundary_conditions,
    const Vector3 & direction, vectorfield & gradient )
{
    // std::cout << "start gradient" << std::endl;
    vectorfield translations = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } };
    auto & n_cells           = geometry.n_cells;

    neighbourfield neigh;

    // TODO: calculate Neighbours outside iterations
    // Neighbours::get_Neighbours(geometry, neigh);

    // TODO: proper usage of neighbours
    // Hardcoded neighbours - for spin current in a rectangular lattice
    neigh = neighbourfield( 0 );
    Neighbour neigh_tmp;
    neigh_tmp.i         = 0;
    neigh_tmp.j         = 0;
    neigh_tmp.idx_shell = 0;

    neigh_tmp.translations[0] = 1;
    neigh_tmp.translations[1] = 0;
    neigh_tmp.translations[2] = 0;
    neigh.push_back( neigh_tmp );

    neigh_tmp.translations[0] = -1;
    neigh_tmp.translations[1] = 0;
    neigh_tmp.translations[2] = 0;
    neigh.push_back( neigh_tmp );

    neigh_tmp.translations[0] = 0;
    neigh_tmp.translations[1] = 1;
    neigh_tmp.translations[2] = 0;
    neigh.push_back( neigh_tmp );

    neigh_tmp.translations[0] = 0;
    neigh_tmp.translations[1] = -1;
    neigh_tmp.translations[2] = 0;
    neigh.push_back( neigh_tmp );

    neigh_tmp.translations[0] = 0;
    neigh_tmp.translations[1] = 0;
    neigh_tmp.translations[2] = 1;
    neigh.push_back( neigh_tmp );

    neigh_tmp.translations[0] = 0;
    neigh_tmp.translations[1] = 0;
    neigh_tmp.translations[2] = -1;
    neigh.push_back( neigh_tmp );

    // Loop over vectorfield
    for( unsigned int ispin = 0; ispin < vf.size(); ++ispin )
    {
        auto translations_i
            = Indexing::translations_from_idx( n_cells, geometry.n_cell_atoms, ispin ); // transVec of spin i
        // int k = i%geometry.n_cell_atoms; // index within unit cell - k=0 for all cases used in the thesis
        scalar n = 0;

        gradient[ispin].setZero();

        std::vector<Vector3> euclidean{ { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
        std::vector<Vector3> contrib = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } };
        Vector3 proj                 = { 0, 0, 0 };
        Vector3 projection_inv       = { 0, 0, 0 };

        // TODO: both loops together.

        // Loop over neighbours of this vector to calculate contributions of finite differences to current direction
        for( unsigned int j = 0; j < neigh.size(); ++j )
        {
            if( Indexing::boundary_conditions_fulfilled(
                    geometry.n_cells, boundary_conditions, translations_i, neigh[j].translations ) )
            {
                // Index of neighbour
                int ineigh = Indexing::idx_from_translations(
                    n_cells, geometry.n_cell_atoms, translations_i, neigh[j].translations );
                if( ineigh >= 0 )
                {
                    auto d = geometry.positions[ineigh] - geometry.positions[ispin];
                    for( int dim = 0; dim < 3; ++dim )
                    {
                        proj[dim] += std::abs( euclidean[dim].dot( d.normalized() ) );
                    }
                }
            }
        }
        for( int dim = 0; dim < 3; ++dim )
        {
            if( std::abs( proj[dim] ) > 1e-10 )
                projection_inv[dim] = 1.0 / proj[dim];
        }
        // Loop over neighbours of this vector to calculate finite differences
        for( unsigned int j = 0; j < neigh.size(); ++j )
        {
            if( Indexing::boundary_conditions_fulfilled(
                    geometry.n_cells, boundary_conditions, translations_i, neigh[j].translations ) )
            {
                // Index of neighbour
                int ineigh = Indexing::idx_from_translations(
                    n_cells, geometry.n_cell_atoms, translations_i, neigh[j].translations );
                if( ineigh >= 0 )
                {
                    auto d = geometry.positions[ineigh] - geometry.positions[ispin];
                    for( int dim = 0; dim < 3; ++dim )
                    {
                        contrib[dim] += euclidean[dim].dot( d ) / d.dot( d ) * ( vf[ineigh] - vf[ispin] );
                    }
                }
            }
        }

        for( int dim = 0; dim < 3; ++dim )
        {
            gradient[ispin] += direction[dim] * projection_inv[dim] * contrib[dim];
        }
    }
}

// Compute the linear index from the lattice position
inline int linear_idx(
    const int ib, int a, int b, int c, const int n_cell_atoms, const int n_cells[3], const int bc[3],
    const bool disable_checks = false )
{
    if( !disable_checks )
    {
        const bool valid_basis = ib >= 0 && ib < n_cell_atoms;
        const bool valid_a     = a >= 0 && a < n_cells[0];
        const bool valid_b     = b >= 0 && b < n_cells[1];
        const bool valid_c     = c >= 0 && c < n_cells[2];

        if( !valid_basis )
        {
            return -1;
        }

        if( !valid_a )
        {
            if( bc[0] )
                a = ( n_cells[0] + ( a % n_cells[0] ) ) % n_cells[0];
            else
                return -1;
        }

        if( !valid_b )
        {
            if( bc[1] )
                b = ( n_cells[1] + ( b % n_cells[1] ) ) % n_cells[1];
            else
                return -1;
        }

        if( !valid_c )
        {
            if( bc[2] )
                c = ( n_cells[2] + ( c % n_cells[2] ) ) % n_cells[2];
            else
                return -1;
        }
    }
    return ib + n_cell_atoms * ( a + n_cells[0] * ( b + n_cells[1] * c ) );
}

void jacobian(
    const vectorfield & vf, const Data::Geometry & geometry, const intfield & boundary_conditions,
    field<Matrix3> & jacobian )
{
    const int _n_cells[3]             = { geometry.n_cells[0], geometry.n_cells[1], geometry.n_cells[2] };
    const int _boundary_conditions[3] = { boundary_conditions[0], boundary_conditions[1], boundary_conditions[2] };

    // 1.) Choose three linearly independent base vectors, which result from lattice translations
    // TODO: depending on the basis, the bravais vectors might not be the best choice
    std::array<std::array<int, 3>, 3> translations = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };

    Vector3 base_1 = geometry.lattice_constant * geometry.bravais_vectors[0],
            base_2 = geometry.lattice_constant * geometry.bravais_vectors[1],
            base_3 = geometry.lattice_constant * geometry.bravais_vectors[2];

    // 2.) Construct the matrix of base vectors
    Matrix3 base_matrix;
    base_matrix.col( 0 ) = base_1;
    base_matrix.col( 1 ) = base_2;
    base_matrix.col( 2 ) = base_3;

    // 3.) Invert the matrix
    const auto inverse_base_matrix = base_matrix.inverse();

    // 4.) Loop over spins
    Matrix3 m_matrix;
#pragma omp parallel for collapse( 4 ) private( m_matrix )
    for( int c = 0; c < geometry.n_cells[2]; c++ )
    {
        for( int b = 0; b < geometry.n_cells[1]; b++ )
        {
            for( int a = 0; a < geometry.n_cells[0]; a++ )
            {
                for( int ib = 0; ib < geometry.n_cell_atoms; ib++ )
                {
                    const auto idx_cur
                        = linear_idx( ib, a, b, c, geometry.n_cell_atoms, _n_cells, _boundary_conditions, true );

                    for( int trans_idx = 0; trans_idx < 3; trans_idx++ )
                    {
                        const auto & trans = translations[trans_idx];

                        // apply translations in positive direction
                        const auto idx0 = linear_idx(
                            ib, a + trans[0], b + trans[1], c + trans[2], geometry.n_cell_atoms, _n_cells,
                            _boundary_conditions );

                        // apply translations in negative direction
                        const auto idx1 = linear_idx(
                            ib, a - trans[0], b - trans[1], c - trans[2], geometry.n_cell_atoms, _n_cells,
                            _boundary_conditions );

                        Vector3 m0 = { 0, 0, 0 };
                        Vector3 m1 = { 0, 0, 0 };

                        scalar factor = 0.5; // Factor 0.5 for central finite differences
                        if( idx0 >= 0 )
                        {
                            m0 = vf[idx0];
                        }
                        else
                        {
                            m0 = vf[idx_cur];
                            factor *= 2; // Increase factor because now only backward difference
                        }

                        if( idx1 >= 0 )
                        {
                            m1 = vf[idx1];
                        }
                        else
                        {
                            m1 = vf[idx_cur];
                            factor *= 2; // Increase factor because now only forward difference
                        }

                        const Vector3 tmp         = factor * ( m0 - m1 );
                        m_matrix.col( trans_idx ) = tmp;
                    }

                    jacobian[idx_cur] = m_matrix * inverse_base_matrix;
                }
            }
        }
    }
}

} // namespace Vectormath

} // namespace Engine
