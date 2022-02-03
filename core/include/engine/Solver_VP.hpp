#include <engine/Backend_par.hpp>

template<>
inline void Method_Solver<Solver::VP>::Initialize()
{
    this->forces         = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->forces_virtual = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );

    this->configurations_temp = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for( int i = 0; i < this->noi; i++ )
        configurations_temp[i] = std::shared_ptr<vectorfield>( new vectorfield( this->nos ) );

    this->velocities = std::vector<vectorfield>( this->noi, vectorfield( this->nos, Vector3::Zero() ) ); // [noi][nos]
    this->velocities_previous = velocities;                                                              // [noi][nos]
    this->forces_previous     = velocities;                                                              // [noi][nos]
    this->projection          = std::vector<scalar>( this->noi, 0 );                                     // [noi]
    this->force_norm2         = std::vector<scalar>( this->noi, 0 );                                     // [noi]
};

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
        auto f    = forces[i].data();
        auto f_pr = forces_previous[i].data();
        auto v    = velocities[i].data();
        auto v_pr = velocities_previous[i].data();

        Backend::par::apply(
            forces[i].size(),
            [f, f_pr, v, v_pr] SPIRIT_LAMBDA( int idx )
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
        auto & velocity   = velocities[i];
        auto & force      = forces[i];
        auto & force_prev = forces_previous[i];

        auto f      = forces[i].data();
        auto f_pr   = forces_previous[i].data();
        auto v      = velocities[i].data();
        auto m_temp = this->m;

        // Calculate the new velocity
        Backend::par::apply(
            force.size(),
            [f, f_pr, v, m_temp] SPIRIT_LAMBDA( int idx ) { v[idx] += 0.5 / m_temp * ( f_pr[idx] + f[idx] ); } );

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
        auto & velocity           = velocities[i];
        auto & force              = forces[i];
        auto & configuration      = *( configurations[i] );
        auto & configuration_temp = *( configurations_temp[i] );

        auto f         = forces[i].data();
        auto v         = velocities[i].data();
        auto conf      = ( configurations[i] )->data();
        auto conf_temp = ( configurations_temp[i] )->data();

        scalar dt    = this->systems[i]->llg_parameters->dt;
        scalar ratio = projection_full / force_norm2_full;
        auto m_temp  = this->m;

        // Calculate the projected velocity
        if( projection_full <= 0 )
        {
            Vectormath::fill( velocity, { 0, 0, 0 } );
        }
        else
        {
            Backend::par::apply( force.size(), [f, v, ratio] SPIRIT_LAMBDA( int idx ) { v[idx] = f[idx] * ratio; } );
        }

        Backend::par::apply(
            force.size(),
            [conf, conf_temp, dt, m_temp, v, f] SPIRIT_LAMBDA( int idx )
            {
                conf_temp[idx] = conf[idx] + dt * v[idx] + 0.5 / m_temp * dt * f[idx];
                conf[idx]      = conf_temp[idx].normalized();
            } );
    }
};

template<>
inline std::string Method_Solver<Solver::VP>::SolverName()
{
    return "VP";
};

template<>
inline std::string Method_Solver<Solver::VP>::SolverFullName()
{
    return "Velocity Projection";
};