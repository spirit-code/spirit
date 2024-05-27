#pragma once

#include "engine/backend/Transform_Iterator.hpp"
#include "engine/backend/Zip_Iterator.hpp"
#include <Spirit/Spirit_Defines.h>
#include <data/Parameters_Method_LLG.hpp>
#include <data/Spin_System.hpp>
#include <engine/Vectormath.hpp>
#include <engine/spin/Method_Solver.hpp>
#include <utility/Constants.hpp>

namespace Engine
{

namespace Common
{

template<Solver solver>
struct Method_LLG
{
    constexpr Method_LLG() noexcept = default;
    constexpr Method_LLG( const int nos )
            : temperature_distribution( nos, 0.0 ), xi( nos, Vector3::Zero() ), jacobians( nos, Matrix3::Zero() ){};

    void Prepare_Thermal_Field( Data::Parameters_Method_LLG & parameters, const Data::Geometry & geometry )
    {
        namespace Constants  = Utility::Constants;
        const auto & damping = parameters.damping;

        if( parameters.temperature <= 0 && parameters.temperature_gradient_inclination == 0 )
            return;

        // ensure appropriately sized
        if( geometry.nos != xi.size() )
            xi = vectorfield( geometry.nos, Vector3::Zero() );

        const scalar epsilon
            = std::sqrt( 2 * damping * parameters.dt * Constants::gamma / Constants::mu_B * Constants::k_B )
              / ( 1 + damping * damping );

        // PRNG gives Gaussian RN with width 1 -> scale by epsilon and sqrt(T/mu_s)
        auto distribution = std::normal_distribution<scalar>{ 0, 1 };

        // If we have a temperature gradient, we use the distribution (scalarfield)
        if( parameters.temperature_gradient_inclination != 0 )
        {
            // ensure appropriately sized
            if( temperature_distribution.size() != xi.size() )
                temperature_distribution = scalarfield( xi.size(), 0.0 );

            // Calculate distribution
            Vectormath::get_gradient_distribution(
                geometry, parameters.temperature_gradient_direction, parameters.temperature,
                parameters.temperature_gradient_inclination, temperature_distribution, 0, 1e30 );

            // TODO: parallelization of this is actually not quite so trivial
            // #pragma omp parallel for
            for( std::size_t i = 0; i < xi.size(); ++i )
            {
                for( int dim = 0; dim < 3; ++dim )
                    xi[i][dim] = epsilon * std::sqrt( temperature_distribution[i] / geometry.mu_s[i] )
                                 * distribution( parameters.prng );
            }
        }
        // If we only have homogeneous temperature we do it more efficiently
        else if( parameters.temperature > 0 )
        {
            // TODO: parallelization of this is actually not quite so trivial
            // #pragma omp parallel for
            for( std::size_t i = 0; i < xi.size(); ++i )
            {
                for( int dim = 0; dim < 3; ++dim )
                    xi[i][dim] = epsilon * std::sqrt( parameters.temperature / geometry.mu_s[i] )
                                 * distribution( parameters.prng );
            }
        }

        // std::unreachable();
    }

    void Virtual_Force_Spin(
        const Data::Parameters_Method_LLG & parameters, const Data::Geometry & geometry,
        const intfield & boundary_conditions, const vectorfield & image, const vectorfield & force,
        vectorfield & force_virtual )
    {
        //////////
        namespace Constants = Utility::Constants;
        // time steps
        scalar damping = parameters.damping;
        // dt = time_step [ps] * gyromagnetic ratio / mu_B / (1+damping^2) <- not implemented
        scalar dtg     = parameters.dt * Constants::gamma / Constants::mu_B / ( 1 + damping * damping );
        scalar sqrtdtg = dtg / std::sqrt( parameters.dt );
        // STT
        // - monolayer
        scalar a_j      = parameters.stt_magnitude;
        Vector3 s_c_vec = parameters.stt_polarisation_normal;
        // - gradient
        scalar b_j  = a_j;             // pre-factor b_j = u*mu_s/gamma (see bachelorthesis Constantin)
        scalar beta = parameters.beta; // non-adiabatic parameter of correction term
        Vector3 je  = s_c_vec;         // direction of current
        //////////

        // This is the force calculation as it should be for direct minimization
        // TODO: Also calculate force for VP solvers without additional scaling
        if constexpr( solver == Solver::LBFGS_OSO || solver == Solver::LBFGS_Atlas )
        {
            Vectormath::set_c_cross( 1.0, image, force, force_virtual );
        }
        else if( parameters.direct_minimization || solver == Solver::VP || solver == Solver::VP_OSO )
        {
            dtg = parameters.dt * Constants::gamma / Constants::mu_B;
            Vectormath::set_c_cross( dtg, image, force, force_virtual );
        }
        // Dynamics simulation
        else
        {
            Backend::transform(
                SPIRIT_PAR Backend::make_zip_iterator( force.begin(), image.begin(), geometry.mu_s.begin() ),
                Backend::make_zip_iterator( force.end(), image.end(), geometry.mu_s.end() ), force_virtual.begin(),
                Backend::make_zip_function(
                    [dtg, damping] SPIRIT_LAMBDA( const Vector3 & f, const Vector3 & n, const scalar mu_s ) -> Vector3
                    { return dtg / mu_s * ( f + damping * n.cross( f ) ); } ) );

            // STT
            if( a_j > 0 )
            {
                if( parameters.stt_use_gradient )
                {
                    if( jacobians.size() != geometry.nos )
                        jacobians = field<Matrix3>( geometry.nos, Matrix3::Zero() );

                    // Gradient approximation for in-plane currents
                    Vectormath::jacobian( image, geometry, boundary_conditions, jacobians );

                    // Gradient in current richtung, daher => *(-1)
                    // TODO: a_j durch b_j ersetzen
                    const scalar c1 = dtg * a_j * ( damping - beta );
                    const scalar c2 = dtg * a_j * ( 1 + beta * damping );
                    Backend::for_each_n(
                        SPIRIT_PAR Backend::make_zip_iterator(
                            force_virtual.begin(), image.begin(), jacobians.begin() ),
                        force_virtual.size(),
                        Backend::make_zip_function(
                            [c1, c2, je] SPIRIT_LAMBDA( Vector3 & fv, const Vector3 & n, const Matrix3 & jacobian )
                            {
                                const Vector3 s_c_vec = jacobian * je;
                                fv += c1 * s_c_vec + c2 * s_c_vec.cross( n );
                            } ) );
                }
                else
                {
                    const Vector3 v1 = -dtg * a_j * ( damping - beta ) * s_c_vec;
                    const Vector3 v2 = -dtg * a_j * ( 1 + beta * damping ) * s_c_vec;
                    // Monolayer approximation
                    Backend::for_each_n(
                        SPIRIT_PAR Backend::make_zip_iterator( force_virtual.begin(), image.begin() ),
                        force_virtual.size(),
                        Backend::make_zip_function( [v1, v2] SPIRIT_LAMBDA( Vector3 & fv, const Vector3 & n )
                                                    { fv += v1 + v2.cross( n ); } ) );
                }
            }

            // Temperature
            if( parameters.temperature > 0 || parameters.temperature_gradient_inclination != 0 )
            {
                Backend::for_each_n(
                    SPIRIT_PAR Backend::make_zip_iterator( force_virtual.begin(), xi.begin(), image.begin() ),
                    force_virtual.size(),
                    Backend::make_zip_function(
                        [damping] SPIRIT_LAMBDA( Vector3 & fv, const Vector3 & xi, const Vector3 & n )
                        { fv += xi + damping * n.cross( xi ); } ) );
            }
        }
// Apply Pinning
#ifdef SPIRIT_ENABLE_PINNING
        Vectormath::set_c_a( 1, force_virtual, force_virtual, geometry.mask_unpinned );
#endif // SPIRIT_ENABLE_PINNING
    };

private:
    // cache values
    scalarfield temperature_distribution;
    vectorfield xi;

    field<Matrix3> jacobians;
};

} // namespace Common

} // namespace Engine
