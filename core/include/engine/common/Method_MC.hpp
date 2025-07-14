#pragma once
#ifndef SPIRIT_CORE_ENGINE_COMMON_METHOD_MC_HPP
#define SPIRIT_CORE_ENGINE_COMMON_METHOD_MC_HPP

#include <Spirit/Simulation.h>
#include <Spirit/Spirit_Defines.h>
#include <data/Parameters_Method_MC.hpp>
#include <data/Spin_System.hpp>
#include <utility/Exception.hpp>

namespace Engine
{

namespace Common
{

namespace Metropolis
{

template<typename Distribution, typename RandomFunc>
auto step_cone( Distribution && distribution, RandomFunc && prng, const Vector3 & spin, const scalar cone_angle )
    -> Vector3
{
    Matrix3 local_basis;
    // Calculate local basis for the spin
    if( spin.z() < 1 - 1e-10 )
    {
        local_basis.col( 2 ) = spin;
        local_basis.col( 0 ) = -Vector3{ 0, 0, 1 }.cross( spin ).normalized();
        local_basis.col( 1 ) = -spin.cross( local_basis.col( 0 ) );
    }
    else
    {
        local_basis = Matrix3::Identity();
    }

    // Rotation angle between 0 and cone_angle degrees
    const scalar costheta = 1 - ( 1 - std::cos( cone_angle ) ) * distribution( prng );

    const scalar sintheta = std::sqrt( 1 - costheta * costheta );

    // Random distribution of phi between 0 and 360 degrees
    const scalar phi = 2 * Utility::Constants::Pi * distribution( prng );

    // New spin orientation in local basis
    Vector3 local_spin_new{ sintheta * std::cos( phi ), sintheta * std::sin( phi ), costheta };

    // New spin orientation in regular basis
    return local_basis * local_spin_new;
};

template<typename Distribution, typename RandomFunc>
auto step_sphere( Distribution && distribution, RandomFunc && prng ) -> Vector3
{
    // Rotation angle between 0 and 180 degrees
    const scalar costheta = 1 - 2 * distribution( prng );

    const scalar sintheta = std::sqrt( 1 - costheta * costheta );

    // Random distribution of phi between 0 and 360 degrees
    const scalar phi = 2 * Utility::Constants::Pi * distribution( prng );

    // New spin orientation in local basis
    return { sintheta * std::cos( phi ), sintheta * std::sin( phi ), costheta };
};

// semi-classical trial step choosing from level sets along a quantization axis
// NOTE: The implementation assumes that the quantization axis does not depend on the orientation of the spin that is
// changed here. That means it only works if the energy depends linearly on the spin orientation (Zeeman-like).
template<typename Distribution, typename RandomFunc, typename IntDistribution>
auto step_semi_classical_zeeman(
    Distribution && distribution, RandomFunc && prng, IntDistribution && distribution_int,
    const Vector3 & axis ) -> Vector3
{
    // maps the integer range [a, b] produced by the int distribution to equidistant values in [-1, 1]
    const scalar costheta
        = static_cast<scalar>( 2 * distribution_int( prng ) - ( distribution_int.a() + distribution_int.b() ) )
          / ( distribution_int.b() - distribution_int.a() );

    const scalar sintheta = std::sqrt( 1 - costheta * costheta );

    // Random distribution of phi between 0 and 360 degrees
    const scalar phi = 2 * Utility::Constants::Pi * distribution( prng );

    // New spin orientation in regular basis
    return Vectormath::dreibein( axis ) * Vector3{ sintheta * std::cos( phi ), sintheta * std::sin( phi ), costheta };
};

// Simple metropolis trial step
template<typename Hamiltonian, typename StateType, typename SharedData>
void trial_spin( const int idx, StateType & state, Hamiltonian & hamiltonian, SharedData & shared )
{
    const auto & geometry = hamiltonian.get_geometry();

    int ispin;
    if( shared.parameters_mc.metropolis_random_sample )
        // Better statistics, but additional calculation of random number
        ispin = shared.distribution_idx( shared.parameters_mc.prng );
    else
        // Faster, but worse statistics
        ispin = idx;

    if( !Indexing::check_atom_type( geometry.atom_types[ispin] ) )
        return;

    const Vector3 spin_pre  = state.spin[ispin];
    const Vector3 spin_post = [&shared, &spin_pre, &geometry, ispin, &state, &hamiltonian]
    {
        switch( shared.parameters_mc.metropolis_step )
        {
            case Data::Metropolis_Step::CONE: // Sample a cone
                return Metropolis::step_cone(
                    shared.distribution, shared.parameters_mc.prng, spin_pre, shared.cone_angle );
            case Data::Metropolis_Step::SPHERE: // Sample the entire unit sphere
                return Metropolis::step_sphere( shared.distribution, shared.parameters_mc.prng );
            case Data::Metropolis_Step::SEMI_CLASSICAL: // Sample semi-classically from quantized level sets
                return Metropolis::step_semi_classical_zeeman(
                    shared.distribution, shared.parameters_mc.prng,
                    /*distribution_int=*/std::uniform_int_distribution<int>( 0, geometry.spin_qn[ispin] ),
                    /*axis=*/hamiltonian.Spin_Gradient_Local( ispin, state ).normalized() );
            default:
                spirit_throw(
                    Utility::Exception_Classifier::Unknown_Solver, Utility::Log_Level::Error,
                    fmt::format(
                        "Unsupported metropolis step '{}'!",
                        Utility::Enum::name( shared.parameters_mc.metropolis_step ) ) );
        };
    }();

    // Energy difference of configurations with and without displacement
    const scalar E_pre  = hamiltonian.Energy_Single_Spin( ispin, state );
    state.spin[ispin]   = spin_post;
    const scalar E_post = hamiltonian.Energy_Single_Spin( ispin, state );
    const scalar Ediff  = E_post - E_pre;

    // Metropolis criterion: accept the step if the energy fell
    if( Ediff <= 1e-14 )
        return;

    // potentially reject the step if energy rose
    if( shared.parameters_mc.temperature < 1e-12 )
    {
        // Restore the spin
        state.spin[ispin] = spin_pre;
        // Counter for the number of rejections
        ++shared.n_rejected;
    }
    else
    {
        // Exponential factor
        const scalar exp_ediff = std::exp( -Ediff * shared.beta );
        // Metropolis random number
        const scalar x_metropolis = shared.distribution( shared.parameters_mc.prng );

        // Only reject if random number is larger than exponential
        if( exp_ediff < x_metropolis )
        {
            // Restore the spin
            state.spin[ispin] = spin_pre;
            // Counter for the number of rejections
            ++shared.n_rejected;
        }
    }
}

/* Directon Constrained Metropolis Monte Carlo trial step following:
 * https://link.aps.org/doi/10.1103/PhysRevB.82.054415
 * This algorithm extends the one described in the paper by also taking
 * a variable `mu_s` into account.
 */
template<typename Hamiltonian, typename StateType, typename SharedData>
void trial_spin_magnetization_constrained(
    const int idx, StateType & state, Hamiltonian & hamiltonian, SharedData & shared )
{
    const auto & geometry = hamiltonian.get_geometry();
    const int nos         = state.spin.size();

    // changed spin and counteracting spin
    int ispin, jspin;
    if( shared.parameters_mc.metropolis_random_sample )
    {
        // Better statistics, but additional calculation of random number
        ispin = shared.distribution_idx( shared.parameters_mc.prng );
        if( !Indexing::check_atom_type( geometry.atom_types[ispin] ) )
            return;

        // try to find a second spin to compensate
        {
            const int max_retry = std::max( nos, 10 );
            int try_idx         = 0;
            for( ; try_idx < max_retry; ++try_idx )
            {
                jspin = shared.distribution_idx( shared.parameters_mc.prng );
                if( jspin != ispin && Indexing::check_atom_type( geometry.atom_types[jspin] ) )
                    break;
            }
            if( try_idx >= max_retry )
                spirit_throw(
                    Utility::Exception_Classifier::Unknown_Exception, Utility::Log_Level::Error,
                    "Cannot find a random second compensation spin! (too many retries)" );
        }
    }
    else
    {
        // Faster, but much worse statistics
        ispin = idx;
        if( !Indexing::check_atom_type( geometry.atom_types[ispin] ) )
            return;
        const int max_retry = nos;
        int try_idx         = 0;
        jspin               = ( idx + nos / 2 ) % nos;
        for( ; try_idx < max_retry; ++try_idx )
        {
            if( jspin != ispin && Indexing::check_atom_type( geometry.atom_types[jspin] ) )
                break;
            jspin = ( jspin + 1 ) % nos;
        }
        if( try_idx >= max_retry )
            spirit_throw(
                Utility::Exception_Classifier::Unknown_Exception, Utility::Log_Level::Error,
                "Cannot find a second compensation spin!" );
    }

    // At this point both chosen atoms should have a valid atom type
    // NOTE: Should atoms of different type serve as compensation for each other?
    assert(
        Indexing::check_atom_type( geometry.atom_types[ispin] )
        && Indexing::check_atom_type( geometry.atom_types[jspin] ) );

    const scalar mu_s_ratio = geometry.mu_s[ispin] / geometry.mu_s[jspin];
    const auto spin_i_pre   = state.spin[ispin];
    const auto spin_j_pre   = state.spin[jspin];
    const auto spin_i_post  = [&shared, &spin_i_pre]
    {
        switch( shared.parameters_mc.metropolis_step )
        {
            case Data::Metropolis_Step::CONE: // Sample a cone
                return Metropolis::step_cone(
                    shared.distribution, shared.parameters_mc.prng, spin_i_pre, shared.cone_angle );
            case Data::Metropolis_Step::SPHERE: // Sample the entire unit sphere
                return Metropolis::step_sphere( shared.distribution, shared.parameters_mc.prng );
            default:
                spirit_throw(
                    Utility::Exception_Classifier::Unknown_Solver, Utility::Log_Level::Error,
                    fmt::format(
                        "Unsupported metropolis step '{}'!",
                        Utility::Enum::name( shared.parameters_mc.metropolis_step ) ) );
        };
    }();

    // calculate compensation spin
    Vector3 spin_j_post          = shared.orth_projector * ( spin_j_pre + mu_s_ratio * ( spin_i_pre - spin_i_post ) );
    const scalar sz_post_squared = 1 - spin_j_post.squaredNorm();
    if( sz_post_squared < 0 )
    {
        ++shared.n_rejected;
        return;
    }

    const scalar sz_pre  = shared.para_projector.dot( spin_j_pre );
    const scalar sz_post = std::copysign( std::sqrt( sz_post_squared ), sz_pre );
    spin_j_post += sz_post * shared.para_projector;

    // calculate magnetization
    const scalar magnetization_post = shared.magnetization_pre
                                      + ( geometry.mu_s[ispin] * shared.para_projector.dot( spin_i_post - spin_i_pre )
                                          + geometry.mu_s[jspin] * ( sz_post - sz_pre ) )
                                            / static_cast<scalar>( nos );

    if( magnetization_post < 0 )
    {
        ++shared.n_rejected;
        return;
    }

    // Energy difference of configurations with and without displacement
    // The `Energy_Single_Spin` method assumes changes to a single spin.
    // To determine the change in energy from the full change correctly
    // we have to evaluate the differences from each change individually.
    const scalar Ediff = [ispin, jspin, &spin_i_post, &spin_j_post, &state, &hamiltonian]
    {
        scalar Ediff      = -1.0 * hamiltonian.Energy_Single_Spin( ispin, state );
        state.spin[ispin] = spin_i_post;
        Ediff += hamiltonian.Energy_Single_Spin( ispin, state ) - hamiltonian.Energy_Single_Spin( jspin, state );
        state.spin[jspin] = spin_j_post;
        Ediff += hamiltonian.Energy_Single_Spin( jspin, state );
        return Ediff;
    }();

    const scalar jacobian_pre  = shared.magnetization_pre * shared.magnetization_pre / sz_pre;
    const scalar jacobian_post = magnetization_post * magnetization_post / sz_post;
    const scalar metropolis_pb = std::abs( jacobian_post / jacobian_pre ) * std::exp( -Ediff * shared.beta );

    // Metropolis criterion: reject the step with probability (1 - min(1, metropolis_pb))
    if( ( metropolis_pb < 1e-14 )
        || ( metropolis_pb < 1.0 && metropolis_pb < shared.distribution( shared.parameters_mc.prng ) ) )
    {
        // Restore the spin
        state.spin[ispin] = spin_i_pre;
        state.spin[jspin] = spin_j_pre;
        // Counter for the number of rejections
        ++shared.n_rejected;
        return;
    }

    // finally accept the step and update the current magnetization
    shared.magnetization_pre = magnetization_post;
}

} // namespace Metropolis

} // namespace Common

} // namespace Engine

#endif
