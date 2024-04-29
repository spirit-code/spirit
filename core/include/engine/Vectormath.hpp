#pragma once

#include <data/Geometry.hpp>
#include <engine/Vectormath_Defines.hpp>

#include <Eigen/Core>

#include <random>
#include <vector>

namespace Engine
{

namespace Vectormath
{

// A "rotated view" into a vectorfield, with optional shifts applied before and after rotation.
template<class vectorfield_t, class vector_t>
class Rotated_View;

// Constructs a rotation matrix that rotates to a frame with "normal" as the z-axis
Matrix3 dreibein( const Vector3 & normal );

/////////////////////////////////////////////////////////////////
//////// Single Vector Math

// Angle between two vectors, assuming both are normalized
SPIRIT_HOSTDEVICE scalar angle( const Vector3 & v1, const Vector3 & v2 );
// Rotate a vector around an axis by a certain degree (Implemented with Rodrigue's formula)
void rotate( const Vector3 & v, const Vector3 & axis, const scalar & angle, Vector3 & v_out );
void rotate( const vectorfield & v, const vectorfield & axis, const scalarfield & angle, vectorfield & v_out );

// Decompose a vector into numbers of translations in a basis
Vector3 decompose( const Vector3 & v, const std::vector<Vector3> & basis );

/////////////////////////////////////////////////////////////////
//////// Vectorfield Math - special stuff

// Calculate the mean of a vectorfield
Vector3 Magnetization( const vectorfield & vf, const scalarfield & mu_s );

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
template<typename T, typename U = T>
void fill( field<T> & vector, const U & value );

// TODO: Add the test
template<typename T, typename U = T>
void fill( field<T> & vector, const U & value, const intfield & mask );

// Scale a field by a given value
template<typename T>
void scale( field<T> & vector, scalar value );

// Scale a field by a scalarfield
template<typename T>
void scale( field<T> & vector, const scalarfield & sf );

// Add a scalar to all entries of a scalarfield
template<typename T, typename U = T>
void add( field<T> & vector, const U & value );

// Sum over a scalarfield
template<typename T>
T sum( const field<T> & vector );

// Calculate the mean of a scalarfield
template<typename T>
T mean( const field<T> & vector );

// Cut off all values to remain in a certain range
void set_range( scalarfield & sf, scalar sf_min, scalar sf_max );

template<typename T>
void divide( field<T> & vector, scalar s );

template<typename T>
void divide( field<T> & numerator, const scalarfield & denominator );

// divide two scalarfields
template<typename T, typename U>
void divide( const field<T> & numerator, const scalarfield & denominator, field<U> & out );

// sets vf := v
// vf is a vectorfield
// v is a vector

// Normalize the vectors of a vectorfield
template<typename V>
void normalize_vectors( field<V> & vector );

// Get the norm of a vectorfield
template<typename V>
void norm( const field<V> & vector, scalarfield & norm );

// Maximum norm of a vectorfield
template<typename V>
scalar max_norm( const field<V> & vector );

// TODO: move this function to manifold??
// computes the inner product of two vectorfields v1 and v2
template<typename T>
scalar dot( const field<T> & vf1, const field<T> & vf2 );

// computes the inner products of vectors in v1 and v2
// v1 and v2 are vectorfields
template<typename T>
void dot( const field<T> & vf1, const field<T> & vf2, scalarfield & out );

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

// finite difference implementation for gradients
template<typename StateType, typename GradientType, typename EnergyFunction>
void Gradient( const StateType & spins, GradientType & gradient, EnergyFunction && energy, scalar delta = 1e-3 );

// finite difference implementation for hessians
template<typename StateType, typename HessianType, typename GradientFunction>
void Hessian( const StateType & spins, HessianType & hessian, GradientFunction && gradient, scalar delta = 1e-3 );

} // namespace Vectormath

} // namespace Engine

#include <engine/Vectormath.inl>
