#pragma once
#ifndef SPIRIT_CORE_ENGINE_VECTORMATH_HPP
#define SPIRIT_CORE_ENGINE_VECTORMATH_HPP

#include <memory>
#include <vector>

#include <Eigen/Core>

#include <data/Geometry.hpp>
#include <data/Spin_System.hpp>
#include <engine/Vectormath_Defines.hpp>

namespace Spirit::Engine::Vectormath
{

// A "rotated view" into a vectorfield, with optional shifts applied before and after rotation.
template<class vectorfield_t, class vector_t>
class Rotated_View
{
private:
    const vectorfield_t & field;
    const Matrix3 & rotation_matrix;
    const vector_t & shift_pre;
    const vector_t & shift_post;

public:
    Rotated_View(
        const vectorfield_t & field, const Matrix3 & rotation_matrix, const vector_t & shift_pre = vector_t{ 0, 0, 0 },
        const vector_t & shift_post = vector_t{ 0, 0, 0 } )
            : field( field ), rotation_matrix( rotation_matrix ), shift_pre( shift_pre ), shift_post( shift_post )
    {
    }

    const vector_t operator[]( int idx )
    {
        return rotation_matrix * ( field[idx] - shift_pre ) + shift_post;
    }
};

// Constructs a rotation matrix that rotates to a frame with "normal" as the z-axis
Matrix3 dreibein( const Vector3 & normal );

/////////////////////////////////////////////////////////////////
//////// Single Vector Math

// Angle between two vectors, assuming both are normalized
scalar angle( const Vector3 & v1, const Vector3 & v2 );
// Rotate a vector around an axis by a certain degree (Implemented with Rodrigue's formula)
void rotate( const Vector3 & v, const Vector3 & axis, const scalar & angle, Vector3 & v_out );
void rotate( const vectorfield & v, const vectorfield & axis, const scalarfield & angle, vectorfield & v_out );

// Decompose a vector into numbers of translations in a basis
Vector3 decompose( const Vector3 & v, const std::vector<Vector3> & basis );

/////////////////////////////////////////////////////////////////
//////// Vectorfield Math - special stuff

// Calculate the mean of a vectorfield
std::array<scalar, 3> Magnetization( const vectorfield & vf, const scalarfield & mu_s );

// Calculate the topological charge density inside a vectorfield
void TopologicalChargeDensity(
    const vectorfield & vf, const Data::Geometry & geometry, const intfield & boundary_conditions,
    scalarfield & charge_density, std::vector<int> & triangle_indices );

// Calculate the topological charge inside a vectorfield
scalar TopologicalCharge( const vectorfield & vf, const Data::Geometry & geom, const intfield & boundary_conditions );

void get_random_vector( std::uniform_real_distribution<scalar> & distribution, std::mt19937 & prng, Vector3 & vec );
void get_random_vectorfield( std::mt19937 & prng, vectorfield & xi );
void get_random_vector_unitsphere(
    std::uniform_real_distribution<scalar> & distribution, std::mt19937 & prng, Vector3 & vec );
void get_random_vectorfield_unitsphere( std::mt19937 & prng, vectorfield & xi );

// Calculate a gradient scalar distribution according to a starting value, direction and inclination
void get_gradient_distribution(
    const Data::Geometry & geometry, Vector3 gradient_direction, scalar gradient_start, scalar gradient_inclination,
    scalarfield & distribution, scalar range_min, scalar range_max );

// Calculate the spatial gradient of a vectorfield in a certain direction.
//      This requires to know the underlying geometry, as well as the boundary conditions.
// NOTE: This implementation is only applicable to rectangular geometries.
void directional_gradient(
    const vectorfield & vf, const Data::Geometry & geometry, const intfield & boundary_conditions,
    const Vector3 & direction, vectorfield & gradient );

// Calculate the jacobians of a vectorfield
void jacobian(
    const vectorfield & vf, const Data::Geometry & geometry, const intfield & boundary_conditions,
    field<Matrix3> & jacobian );

/////////////////////////////////////////////////////////////////
//////// Vectormath-like operations

// sets sf := s
// sf is a scalarfield
// s is a scalar
void fill( scalarfield & sf, scalar s );

// TODO: Add the test
void fill( scalarfield & sf, scalar s, const intfield & mask );

// Scale a scalarfield by a given value
void scale( scalarfield & sf, scalar s );

// Add a scalar to all entries of a scalarfield
void add( scalarfield & sf, scalar s );

// Sum over a scalarfield
scalar sum( const scalarfield & sf );

// Calculate the mean of a scalarfield
scalar mean( const scalarfield & sf );

// Cut off all values to remain in a certain range
void set_range( scalarfield & sf, scalar sf_min, scalar sf_max );

// sets vf := v
// vf is a vectorfield
// v is a vector
void fill( vectorfield & vf, const Vector3 & v );
void fill( vectorfield & vf, const Vector3 & v, const intfield & mask );

// Normalize the vectors of a vectorfield
void normalize_vectors( vectorfield & vf );

// Get the norm of a vectorfield
void norm( const vectorfield & vf, scalarfield & norm );

// Maximum absolute component of a vectorfield
scalar max_abs_component( const vectorfield & vf );

// Maximum norm of a vectorfield
scalar max_norm( const vectorfield & vf );

// Scale a vectorfield by a given value
void scale( vectorfield & vf, const scalar & sc );

// Scale a vectorfield by a scalarfield or its inverse
void scale( vectorfield & vf, const scalarfield & sf, bool inverse = false );

// Sum over a vectorfield
Vector3 sum( const vectorfield & vf );

// Calculate the mean of a vectorfield
Vector3 mean( const vectorfield & vf );

// divide two scalarfields
void divide( const scalarfield & numerator, const scalarfield & denominator, scalarfield & out );

// TODO: move this function to manifold??
// computes the inner product of two vectorfields v1 and v2
scalar dot( const vectorfield & vf1, const vectorfield & vf2 );

// computes the inner products of vectors in v1 and v2
// v1 and v2 are vectorfields
void dot( const vectorfield & vf1, const vectorfield & vf2, scalarfield & out );

// TODO: find a more appropriate name
// computes the product of scalars in sf1 and sf2
// sf1 and sf2 are vectorfields
void dot( const scalarfield & sf1, const scalarfield & sf2, scalarfield & out );

// computes the vector (cross) products of vectors in v1 and v2
// v1 and v2 are vector fields
void cross( const vectorfield & vf1, const vectorfield & vf2, vectorfield & out );

// out[i] += c*a
void add_c_a( const scalar & c, const Vector3 & a, vectorfield & out );
// out[i] += c*a[i]
void add_c_a( const scalar & c, const vectorfield & vf, vectorfield & out );
void add_c_a( const scalar & c, const vectorfield & vf, vectorfield & out, const intfield & mask );
// out[i] += c[i]*a[i]
void add_c_a( const scalarfield & c, const vectorfield & vf, vectorfield & out );

// out[i] = c*a
void set_c_a( const scalar & c, const Vector3 & a, vectorfield & out );
void set_c_a( const scalar & c, const Vector3 & a, vectorfield & out, const intfield & mask );
// out[i] = c*a[i]
void set_c_a( const scalar & c, const vectorfield & vf, vectorfield & out );
void set_c_a( const scalar & c, const vectorfield & vf, vectorfield & out, const intfield & mask );
// out[i] = c[i]*a[i]
void set_c_a( const scalarfield & sf, const vectorfield & vf, vectorfield & out );

// out[i] += c * a*b[i]
void add_c_dot( const scalar & c, const Vector3 & a, const vectorfield & b, scalarfield & out );
// out[i] += c * a[i]*b[i]
void add_c_dot( const scalar & c, const vectorfield & a, const vectorfield & b, scalarfield & out );

// out[i] = c * a*b[i]
void set_c_dot( const scalar & c, const Vector3 & a, const vectorfield & b, scalarfield & out );
// out[i] = c * a[i]*b[i]
void set_c_dot( const scalar & c, const vectorfield & a, const vectorfield & b, scalarfield & out );

// out[i] += c * a x b[i]
void add_c_cross( const scalar & c, const Vector3 & a, const vectorfield & b, vectorfield & out );
// out[i] += c * a[i] x b[i]
void add_c_cross( const scalar & c, const vectorfield & a, const vectorfield & b, vectorfield & out );
// out[i] += c[i] * a[i] x b[i]
void add_c_cross( const scalarfield & c, const vectorfield & a, const vectorfield & b, vectorfield & out );

// out[i] = c * a x b[i]
void set_c_cross( const scalar & c, const Vector3 & a, const vectorfield & b, vectorfield & out );
// out[i] = c * a[i] x b[i]
void set_c_cross( const scalar & c, const vectorfield & a, const vectorfield & b, vectorfield & out );

} // namespace Spirit::Engine::Vectormath

#endif