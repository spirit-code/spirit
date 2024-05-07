#pragma once
#include <engine/spin/Method_Solver.hpp>

namespace Engine
{

namespace Spin
{


template<>
class SolverData<Solver::VP> : public SolverMethods
{
protected:
    using SolverMethods::SolverMethods;
    using SolverMethods::Prepare_Thermal_Field;
    using SolverMethods::Calculate_Force;
    using SolverMethods::Calculate_Force_Virtual;
    //////////// VP ///////////////////////////////////////////////////////////////
    // "Mass of our particle" which we accelerate
    static constexpr scalar mass = 1.0;

    // Force in previous step [noi][nos]
    std::vector<vectorfield> forces_previous;
    // Velocity in previous step [noi][nos]
    std::vector<vectorfield> velocities_previous;
    // Velocity used in the Steps [noi][nos]
    std::vector<vectorfield> velocities;
    // Projection of velocities onto the forces [noi]
    std::vector<scalar> projection;
    // |force|^2
    std::vector<scalar> force_norm2;

    std::vector<std::shared_ptr<vectorfield>> configurations_temp;
};

template<>
inline void Method_Solver<Solver::VP>::Initialize()
{
    this->forces         = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->forces_virtual = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );

    this->configurations_temp = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for( int i = 0; i < this->noi; i++ )
        configurations_temp[i] = std::make_shared<vectorfield>( this->nos );

    this->velocities = std::vector<vectorfield>( this->noi, vectorfield( this->nos, Vector3::Zero() ) ); // [noi][nos]
    this->velocities_previous = velocities;                                                              // [noi][nos]
    this->forces_previous     = velocities;                                                              // [noi][nos]
    this->projection          = std::vector<scalar>( this->noi, 0 );                                     // [noi]
    this->force_norm2         = std::vector<scalar>( this->noi, 0 );                                     // [noi]
}

/*
    Template instantiation of the Simulation class for use with the VP Solver.
        The velocity projection method is often efficient for direct minimization,
        but deals poorly with quickly varying fields or stochastic noise.
    Paper: P. F. Bessarab et al., Method for finding mechanism and activation energy
           of magnetic transitions, applied to skyrmion and antivortex annihilation,
           Comp. Phys. Comm. 196, 335 (2015).
*/
template<>
inline void Method_Solver<Solver::VP>::Iteration()
{
    scalar projection_full  = 0;
    scalar force_norm2_full = 0;

    // Set previous
    for( int i = 0; i < noi; ++i )
    {
        const auto * f = raw_pointer_cast( forces[i].data() );
        auto * f_pr    = raw_pointer_cast( forces_previous[i].data() );
        const auto * v = raw_pointer_cast( velocities[i].data() );
        auto * v_pr    = raw_pointer_cast( velocities_previous[i].data() );

        Backend::for_each_n(
            SPIRIT_PAR Backend::make_counting_iterator( 0 ), forces[i].size(),
            [f, f_pr, v, v_pr] SPIRIT_LAMBDA( const int idx )
            {
                f_pr[idx] = f[idx];
                v_pr[idx] = v[idx];
            } );
    }

    // Get the forces on the configurations
    this->Calculate_Force( configurations, forces );
    this->Calculate_Force_Virtual( configurations, forces, forces_virtual );

    for( int i = 0; i < noi; ++i )
    {
        auto & velocity = velocities[i];
        auto & force    = forces[i];

        const auto * f    = raw_pointer_cast( forces[i].data() );
        const auto * f_pr = raw_pointer_cast( forces_previous[i].data() );
        auto * v          = raw_pointer_cast( velocities[i].data() );

        // Calculate the new velocity
        Backend::for_each_n(
            SPIRIT_PAR Backend::make_counting_iterator( 0 ), force.size(),
            [f, f_pr, v] SPIRIT_LAMBDA( const int idx )
            { v[idx] += 0.5 / mass * ( f_pr[idx] + f[idx] ); } );

        // Get the projection of the velocity on the force
        projection[i]  = Vectormath::dot( velocity, force );
        force_norm2[i] = Vectormath::dot( force, force );
    }
    for( int i = 0; i < noi; ++i )
    {
        projection_full += projection[i];
        force_norm2_full += force_norm2[i];
    }
    for( int i = 0; i < noi; ++i )
    {
        auto & velocity = velocities[i];
        auto & force    = forces[i];

        const auto * f   = raw_pointer_cast( forces[i].data() );
        auto * v         = raw_pointer_cast( velocities[i].data() );
        auto * conf      = raw_pointer_cast( ( configurations[i] )->data() );
        auto * conf_temp = raw_pointer_cast( ( configurations_temp[i] )->data() );

        const scalar dt    = this->systems[i]->llg_parameters->dt;
        const scalar ratio = projection_full / force_norm2_full;

        // Calculate the projected velocity
        if( projection_full <= 0 )
        {
            Vectormath::fill( velocity, { 0, 0, 0 } );
        }
        else
        {
            Backend::for_each_n(
                SPIRIT_PAR Backend::make_counting_iterator( 0 ), force.size(),
                [f, v, ratio] SPIRIT_LAMBDA( const int idx ) { v[idx] = f[idx] * ratio; } );
        }

        Backend::for_each_n(
            SPIRIT_PAR Backend::make_counting_iterator( 0 ), force.size(),
            [conf, conf_temp, dt,  v, f] SPIRIT_LAMBDA( const int idx )
            {
                conf_temp[idx] = conf[idx] + dt * v[idx] + 0.5 / mass * dt * f[idx];
                conf[idx]      = conf_temp[idx].normalized();
            } );
    }
}

template<>
inline std::string Method_Solver<Solver::VP>::SolverName()
{
    return "VP";
}

template<>
inline std::string Method_Solver<Solver::VP>::SolverFullName()
{
    return "Velocity Projection";
}

} // namespace Spin

} // namespace Engine
