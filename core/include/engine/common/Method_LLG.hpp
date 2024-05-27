#pragma once

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
            : temperature_distribution( nos, 0.0 ),
              xi( nos, Vector3::Zero() ),
              jacobians( nos, Matrix3::Zero() ),
              s_c_grad( nos, Vector3::Zero() ){};

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
            Vectormath::set_c_a( dtg, force, force_virtual );
            Vectormath::add_c_cross( dtg * damping, image, force, force_virtual );
            Vectormath::divide( force_virtual, geometry.mu_s );

            // STT
            if( a_j > 0 )
            {
                if( parameters.stt_use_gradient )
                {
                    if( jacobians.size() != geometry.nos )
                        jacobians = field<Matrix3>(geometry.nos, Matrix3::Zero());

                    if( s_c_grad.size() != geometry.nos )
                        s_c_grad = vectorfield(geometry.nos, Vector3::Zero());

                    // Gradient approximation for in-plane currents
                    Vectormath::jacobian( image, geometry, boundary_conditions, jacobians );

                    // TODO: merge these operations to eliminate the use of `s_c_grad`
                    Backend::transform(
                        SPIRIT_PAR begin( jacobians ), end( jacobians ), begin( s_c_grad ),
                        [je] SPIRIT_LAMBDA( const Matrix3 & jacobian ) { return jacobian * je; } );

                    Vectormath::add_c_a(
                        dtg * a_j * ( damping - beta ), s_c_grad, force_virtual ); // TODO: a_j durch b_j ersetzen
                    Vectormath::add_c_cross(
                        dtg * a_j * ( 1 + beta * damping ), s_c_grad, image,
                        force_virtual ); // TODO: a_j durch b_j ersetzen
                    // Gradient in current richtung, daher => *(-1)
                }
                else
                {
                    // Monolayer approximation
                    Vectormath::add_c_a( -dtg * a_j * ( damping - beta ), s_c_vec, force_virtual );
                    Vectormath::add_c_cross( -dtg * a_j * ( 1 + beta * damping ), s_c_vec, image, force_virtual );
                }
            }

            // Temperature
            if( parameters.temperature > 0 || parameters.temperature_gradient_inclination != 0 )
            {
                Vectormath::add_c_a( 1, xi, force_virtual );
                Vectormath::add_c_cross( damping, image, xi, force_virtual );
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
    vectorfield s_c_grad;
};

} // namespace Common

} // namespace Engine
