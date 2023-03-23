#pragma once
#include <limits>
#ifndef SPIRIT_CORE_ENGINE_LINESEARCH_HPP
#define SPIRIT_CORE_ENGINE_LINESEARCH_HPP

#include <fmt/format.h>
#include <Eigen/Geometry>
#include <engine/Backend_par.hpp>
#include <engine/Hamiltonian.hpp>
#include <memory>
#include <utility/Exception.hpp>
#include <vector>

#include <iomanip>

namespace Engine
{

template<typename DirectionT>
class Propagator
{
public:
    virtual void propagate( vectorfield & v, const field<DirectionT> & directions, scalar alpha ) = 0;
};

template<typename DirectionT>
class Linesearch
{
protected:
    std::unique_ptr<Propagator<DirectionT>> propagator;

public:
    Linesearch( Propagator<DirectionT> * prop ) : propagator( std::unique_ptr<Propagator<DirectionT>>( prop ) ){};

    virtual void
    run( std::vector<std::shared_ptr<vectorfield>> & configurations, const std::vector<field<DirectionT>> & directions )
        = 0;
};

template<typename DirectionT>
class Trivial_Linesearch : public Linesearch<DirectionT>
{
    // Propagates with step length 1
public:
    Trivial_Linesearch( Propagator<DirectionT> * prop ) : Linesearch<DirectionT>( prop ) {}

    void
    run( std::vector<std::shared_ptr<vectorfield>> & configurations,
         const std::vector<field<DirectionT>> & directions ) override
    {
        for( int img = 0; img < configurations.size(); img++ )
        {
            this->propagator->propagate( *configurations[img], directions[img], 1.0 );
        }
    }
};

class Renormalisation_Propagator : public Propagator<Vector3>
{
public:
    void propagate( vectorfield & v, const field<Vector3> & directions, scalar alpha = 1.0 ) override
    {
        Backend::par::apply( v.size(), [v = v.data(), d = directions.data(), alpha] SPIRIT_LAMBDA( int idx ) {
            v[idx] += alpha * d[idx];
            v[idx].normalize();
        } );
    }
};

class Rotation_Propagator : public Propagator<Vector3>
{
public:
    void propagate( vectorfield & v, const field<Vector3> & directions, scalar alpha = 1.0 ) override
    {
        Backend::par::apply( v.size(), [v = v.data(), d = directions.data(), alpha] SPIRIT_LAMBDA( int idx ) {
            const auto axis  = ( v[idx].cross( d[idx] ) ).normalized();
            const auto angle = alpha * d[idx].norm();
            const auto quat  = Eigen::Quaternion<scalar>( Eigen::AngleAxis<scalar>( angle, axis ) );
            v[idx]           = quat._transformVector( v[idx] );
        } );
    }
};

template<typename DirectionT>
class Quadratic_Backtracking_Linesearch : public Linesearch<DirectionT>
{
private:
    vectorfield gradient_throwaway;
    vectorfield configuration_initial;

    scalarfield energy_start;
    scalarfield energy_end;
    scalar tolerance = 0.01;

    scalarfield a;
    scalarfield b;

    int noi;
    int nos;

    Hamiltonian * hamiltonian;

    void determine_parabola_coefficients(
        std::vector<std::shared_ptr<vectorfield>> & configurations, const std::vector<field<DirectionT>> & directions )
    {

        noi = configurations.size();
        nos = configurations[0]->size();

        if( a.size() != noi )
        {
            a.resize( noi );
            b.resize( noi );
            energy_start.resize( noi );
            energy_end.resize( noi );
        }

        if( gradient_throwaway.size() != nos )
        {
            gradient_throwaway.resize( nos );
            configuration_initial.resize( nos );
        }

        if( hamiltonian == nullptr )
        {
            return;
        }

        // We expect the energy to fit a parabola:
        // E(x) = a * x^2 + b * x + energy_start
        // We determine the parabola from E(0), E'(0) and E'(1) where the search interval is [0,1]
        // Determine a, b and energy_start
        for( int img = 0; img < noi; img++ )
        {
            // Reset energy and gradient to 0
            energy_start[img] = 0;
            Vectormath::fill( gradient_throwaway, Vector3::Zero() );

            hamiltonian->Gradient_and_Energy( *configurations[img], gradient_throwaway, energy_start[img] );
            Manifoldmath::project_tangential( gradient_throwaway, *configurations[img] );
            b[img] = Vectormath::dot( directions[img], gradient_throwaway );
            // b is the projection of the gradient along the tangent

            // Now do the computation at the end of the interval
            // Reset energy and gradient
            energy_end[img] = 0;
            Vectormath::fill( gradient_throwaway, Vector3::Zero() );

            // Copy the old configurations
            configuration_initial = *configurations[img];

            // Propagate to endpoint of interval
            this->propagator->propagate( *configurations[img], directions[img], 1.0 );

            hamiltonian->Gradient_and_Energy( *configurations[img], gradient_throwaway, energy_end[img] );
            Manifoldmath::project_tangential( gradient_throwaway, *configurations[img] );

            // To make comparisons between gradient and direction meaningful, we rotate the gradient back into the
            // tangent frame of the start and then project to its tangent plane
            Backend::par::apply(
                configuration_initial.size(),
                [c_old = configuration_initial.data(), c_new = ( *configurations[img] ).data(),
                 g_new = gradient_throwaway.data()] SPIRIT_LAMBDA( int idx ) {
                    const auto quat = Eigen::Quaternion<scalar>::FromTwoVectors( c_new[idx], c_old[idx] );
                    g_new[idx]      = quat._transformVector( g_new[idx] );
                } );

            a[img] = 0.5 * ( Vectormath::dot( directions[img], gradient_throwaway ) - b[img] );
        }
    }

    bool check_energy( scalar alpha, int img )
    {
        // Compute expected energy difference from parabola coefficients
        const scalar delta_e_expected = b[img] * alpha + a[img] * alpha * alpha;

        // Compute the real energy difference
        const scalar delta_e = energy_end[img] - energy_start[img];

        const scalar kappa = 1e3;
        const scalar criterion_applicability
            = std::abs( kappa * energy_start[img] * std::numeric_limits<scalar>::epsilon() );
        const bool linesearch_applicable = criterion_applicability < tolerance;

        const scalar criterion = std::abs( std::abs( delta_e_expected / delta_e ) - 1.0 );

        if( !linesearch_applicable )
            return true;

        return criterion < tolerance;
    }

public:
    int max_iter = 50;

    Quadratic_Backtracking_Linesearch( Propagator<DirectionT> * prop, Hamiltonian * hamiltonian )
            : Linesearch<DirectionT>( prop ),
              hamiltonian( hamiltonian ),
              a( scalarfield( noi ) ),
              b( scalarfield( noi ) ),
              energy_start( scalarfield( noi ) ),
              energy_end( scalarfield( noi ) ),
              gradient_throwaway( vectorfield( 1 ) )
    {
    }

    virtual void
    run( std::vector<std::shared_ptr<vectorfield>> & configurations,
         const std::vector<field<DirectionT>> & directions ) override
    {
        determine_parabola_coefficients( configurations, directions );
        // Note: Configurations are at end of search interval now

        // For every image check if the prediction of the energy and the actual energy is within tolerance
        for( int img = 0; img < noi; img++ )
        {
            scalar alpha        = 1;
            int iter            = 0;
            bool stop_criterion = false;

            while( !stop_criterion )
            {
                iter++;
                const bool check = check_energy( alpha, img );

                if( check || iter >= max_iter )
                {
                    stop_criterion = true;
                }
                else
                {
                    alpha /= 2.0;
                    this->propagator->propagate(
                        *configurations[img], directions[img], -alpha ); // Note: negative sign is for backtracking

                    Vectormath::fill( gradient_throwaway, Vector3::Zero() );
                    energy_end[img] = 0;
                    hamiltonian->Gradient_and_Energy( *configurations[img], gradient_throwaway, energy_end[img] );
                    Manifoldmath::project_tangential( gradient_throwaway, *configurations[img] );
                }
            }
        }
    }
};

} // namespace Engine

#endif
