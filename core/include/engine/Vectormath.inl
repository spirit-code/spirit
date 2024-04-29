#pragma once

#include <data/Geometry.hpp>
#include <engine/Backend.hpp>
#include <engine/Vectormath_Defines.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

namespace Engine
{

namespace Vectormath
{

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
        const vectorfield_t & field, const Matrix3 & rotation_matrix, const vector_t & shift_pre = vector_t::Zero(),
        const vector_t & shift_post = vector_t::Zero() )
            : field( field ), rotation_matrix( rotation_matrix ), shift_pre( shift_pre ), shift_post( shift_post )
    {
    }

    const vector_t operator[]( int idx )
    {
        return rotation_matrix * ( field[idx] - shift_pre ) + shift_post;
    }
};

SPIRIT_HOSTDEVICE inline scalar angle( const Vector3 & v1, const Vector3 & v2 )
{
    const scalar r = v1.dot( v2 );
    // Angle (clamp to prevent NaNs from occurring)
    return acos( ( 1.0 < r ) ? 1.0 : ( r < -1.0 ) ? -1.0 : r );
}

template<typename T, typename U>
void fill( field<T> & vector, const U & value )
{
    Backend::fill( SPIRIT_PAR vector.begin(), vector.end(), value );
}

// TODO: Add the test
template<typename T, typename U>
void fill( field<T> & vector, const U & value, const intfield & mask )
{
    Backend::transform(
        SPIRIT_PAR mask.begin(), mask.end(), vector.begin(),
        [value] SPIRIT_LAMBDA( const int cond ) -> T { return cond * value; } );
}

// Scale a scalarfield by a given value
template<typename T>
void scale( field<T> & vector, const scalar value )
{
    // NOTE:  using a lambda here generates a segfault in the CUDA release build,
    //        the CUDA debug build works fine though.
    Backend::transform( SPIRIT_PAR vector.begin(), vector.end(), vector.begin(), Backend::scale( value ) );
}

// Scale a vectorfield by a scalarfield or its inverse
template<typename T>
void scale( field<T> & vector, const scalarfield & sf )
{
    Backend::transform(
        SPIRIT_PAR vector.begin(), vector.end(), sf.begin(), vector.begin(),
        [] SPIRIT_LAMBDA( const T & element, const scalar s ) -> T { return element * s; } );
}

// Add a scalar to all entries of a scalarfield
template<typename T, typename U>
void add( field<T> & vector, const U & value )
{
    Backend::transform(
        SPIRIT_PAR vector.begin(), vector.end(), vector.begin(),
        [value] SPIRIT_LAMBDA( const T & element ) -> T { return element + value; } );
}

// Sum over a scalarfield
template<typename T>
T sum( const field<T> & vector )
{
    return Backend::reduce( SPIRIT_PAR vector.begin(), vector.end(), zero_value<T>(), Backend::plus<T>{} );
}

// Calculate the mean of a scalarfield
template<typename T>
T mean( const field<T> & vector )
{
    return Vectormath::sum( vector ) / vector.size();
}

// Cut off all values to remain in a certain range
inline void set_range( scalarfield & sf, scalar sf_min, scalar sf_max )
{
    Backend::transform(
        SPIRIT_PAR sf.begin(), sf.end(), sf.begin(),
        [sf_min, sf_max] SPIRIT_LAMBDA( const scalar value ) -> scalar {
            return ( sf_max < value ) ? sf_max : ( value < sf_min ) ? sf_min : value;
        } );
}

// divide two scalarfields
template<typename T>
void divide( field<T> & vector, const scalar s )
{
    Backend::transform(
        SPIRIT_PAR vector.begin(), vector.end(), vector.begin(),
        [s] SPIRIT_LAMBDA( const T & element ) -> T { return element / s; } );
}

template<typename T>
void divide( field<T> & numerator, const scalarfield & denominator )
{
    Backend::transform(
        SPIRIT_PAR numerator.begin(), numerator.end(), denominator.begin(), numerator.begin(),
        [] SPIRIT_LAMBDA( const T & element, const scalar s ) -> T { return element / s; } );
}

template<typename T, typename U>
void divide( const field<T> & numerator, const scalarfield & denominator, field<U> & out )
{
    Backend::transform(
        SPIRIT_PAR numerator.begin(), numerator.end(), denominator.begin(), out.begin(),
        [] SPIRIT_LAMBDA( const T a, const scalar b ) -> U { return a / b; } );
}

// Normalize the vectors of a vectorfield
template<typename V>
void normalize_vectors( field<V> & vector )
{
    Backend::for_each( SPIRIT_PAR vector.begin(), vector.end(), [] SPIRIT_LAMBDA( V & v ) -> void { v.normalize(); } );
}

// Get the norm of a vectorfield
template<typename V>
void norm( const field<V> & vector, scalarfield & norm )
{
    Backend::transform(
        SPIRIT_PAR vector.begin(), vector.end(), norm.begin(),
        [] SPIRIT_LAMBDA( const V & v ) -> scalar { return v.norm(); } );
}

inline std::pair<scalar, scalar> minmax_component( const vectorfield & v1 )
{
    scalar minval = 1e6, maxval = -1e6;
    std::pair<scalar, scalar> minmax;
#pragma omp parallel for reduction( min : minval ) reduction( max : maxval )
    for( unsigned int i = 0; i < v1.size(); ++i )
    {
        for( int dim = 0; dim < 3; ++dim )
        {
            if( v1[i][dim] < minval )
                minval = v1[i][dim];
            if( v1[i][dim] > maxval )
                maxval = v1[i][dim];
        }
    }
    minmax.first  = minval;
    minmax.second = maxval;
    return minmax;
}

template<typename V>
scalar max_abs_component( const field<V> & vector );

// Maximum absolute component of a vectorfield
template<>
inline scalar max_abs_component( const field<Vector3> & vector )
{
    // We want the Maximum of Absolute Values of all force components on all images
    scalar absmax = 0;
    // Find minimum and maximum values
    std::pair<scalar, scalar> minmax = minmax_component( vector );
    // Mamimum of absolute values
    absmax = std::max( absmax, std::abs( minmax.first ) );
    absmax = std::max( absmax, std::abs( minmax.second ) );
    // Return
    return absmax;
}

// Maximum norm of a vectorfield
template<typename V>
scalar max_norm( const field<V> & vector )
{
    return sqrt( Backend::transform_reduce(
        SPIRIT_PAR vector.begin(), vector.end(), scalar( 0 ),
        [] SPIRIT_LAMBDA( const scalar & lhs, const scalar & rhs ) { return ( lhs < rhs ) ? rhs : lhs; },
        [] SPIRIT_LAMBDA( const V & v ) { return v.squaredNorm(); } ) );
}

// TODO: move this function to manifold??
// computes the inner product of two vectorfields v1 and v2
template<typename T>
scalar dot( const field<T> & vf1, const field<T> & vf2 )
{
    return Backend::transform_reduce(
        SPIRIT_PAR vf1.begin(), vf1.end(), vf2.begin(), scalar( 0 ), Backend::plus<scalar>{}, Backend::dot<T>{} );
}

template<>
inline scalar dot( const field<scalar> & vf1, const field<scalar> & vf2 )
{
    return Backend::transform_reduce(
        SPIRIT_PAR vf1.begin(), vf1.end(), vf2.begin(), scalar( 0 ), Backend::plus<scalar>{},
        Backend::multiplies<scalar>{} );
}

// computes the inner products of vectors in v1 and v2
// v1 and v2 are vectorfields
template<typename T>
void dot( const field<T> & vf1, const field<T> & vf2, scalarfield & out )
{
    Backend::transform( SPIRIT_PAR vf1.begin(), vf1.end(), vf2.begin(), out.begin(), Backend::dot<T>{} );
}

template<>
inline void dot( const field<scalar> & vf1, const field<scalar> & vf2, scalarfield & out )
{
    Backend::transform( SPIRIT_PAR vf1.begin(), vf1.end(), vf2.begin(), out.begin(), Backend::multiplies<scalar>{} );
}

inline void cross( const vectorfield & vf1, const vectorfield & vf2, vectorfield & out )
{
    Backend::transform( SPIRIT_PAR vf1.begin(), vf1.end(), vf2.begin(), out.begin(), Backend::cross<Vector3>{} );
}

template<typename StateType, typename GradientType, typename EnergyFunction>
void Gradient( const StateType & spins, GradientType & gradient, EnergyFunction && energy, scalar delta )
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
}

template<typename StateType, typename HessianType, typename GradientFunction>
void Hessian( const StateType & spins, HessianType & hessian, GradientFunction && gradient, scalar delta )
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
}

} // namespace Vectormath

} // namespace Engine
