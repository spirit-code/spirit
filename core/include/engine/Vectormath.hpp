#pragma once
#ifndef SPIRIT_CORE_ENGINE_VECTORMATH_HPP
#define SPIRIT_CORE_ENGINE_VECTORMATH_HPP

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

// finite difference implementation for gradients
template<typename StateType, typename GradientType, typename EnergyFunction>
void Gradient( const StateType & spins, GradientType & gradient, EnergyFunction && energy, scalar delta = 1e-3 )
{
    static_assert( std::is_convertible_v<decltype( energy( spins ) ), scalar> );

    std::size_t nos = spins.size();

    // Calculate finite difference
    vectorfield spins_plus( nos );
    vectorfield spins_minus( nos );

    spins_plus  = spins;
    spins_minus = spins;

    for( std::size_t i = 0; i < nos; ++i )
    {
#pragma unroll
        for( std::uint8_t dim = 0; dim < 3; ++dim )
        {
            // Displace
            spins_plus[i][dim] += delta;
            spins_minus[i][dim] -= delta;

            // Calculate gradient component
            scalar E_plus    = energy( spins_plus );
            scalar E_minus   = energy( spins_minus );
            gradient[i][dim] = 0.5 * ( E_plus - E_minus ) / delta;

            // Un-Displace
            spins_plus[i][dim] -= delta;
            spins_minus[i][dim] += delta;
        }
    }
};

// finite difference implementation for hessians
template<typename StateType, typename HessianType, typename GradientFunction>
void Hessian( const StateType & spins, HessianType & hessian, GradientFunction && gradient, scalar delta = 1e-3 )
{
    static_assert( std::is_invocable_v<GradientFunction, const StateType &, vectorfield &> );

    // This is a regular finite difference implementation (probably not very efficient)
    // using the differences between gradient values (not function)
    // see https://v8doc.sas.com/sashtml/ormp/chap5/sect28.htm

    std::size_t nos = spins.size();

    vectorfield spins_pi( nos );
    vectorfield spins_mi( nos );
    vectorfield spins_pj( nos );
    vectorfield spins_mj( nos );

    spins_pi = spins;
    spins_mi = spins;
    spins_pj = spins;
    spins_mj = spins;

    vectorfield grad_pi( nos );
    vectorfield grad_mi( nos );
    vectorfield grad_pj( nos );
    vectorfield grad_mj( nos );

    for( std::size_t i = 0; i < nos; ++i )
    {
        for( std::size_t j = 0; j < nos; ++j )
        {
#pragma unroll
            for( std::uint8_t alpha = 0; alpha < 3; ++alpha )
            {
#pragma unroll
                for( std::uint8_t beta = 0; beta < 3; ++beta )
                {
                    // Displace
                    spins_pi[i][alpha] += delta;
                    spins_mi[i][alpha] -= delta;
                    spins_pj[j][beta] += delta;
                    spins_mj[j][beta] -= delta;

                    // Calculate Hessian component
                    gradient( spins_pi, grad_pi );
                    gradient( spins_mi, grad_mi );
                    gradient( spins_pj, grad_pj );
                    gradient( spins_mj, grad_mj );

                    hessian( 3 * i + alpha, 3 * j + beta )
                        = 0.25 / delta
                          * ( grad_pj[i][alpha] - grad_mj[i][alpha] + grad_pi[j][beta] - grad_mi[j][beta] );

                    // Un-Displace
                    spins_pi[i][alpha] -= delta;
                    spins_mi[i][alpha] += delta;
                    spins_pj[j][beta] -= delta;
                    spins_mj[j][beta] += delta;
                }
            }
        }
    }
};

} // namespace Vectormath

} // namespace Engine

#endif
