template<>
inline void Method_Solver<Solver::VP_OSO>::Initialize()
{
    this->forces         = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->forces_virtual = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );

    this->configurations_temp = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for( int i = 0; i < this->noi; i++ )
        configurations_temp[i] = std::shared_ptr<vectorfield>( new vectorfield( this->nos ) );

    this->velocities = std::vector<vectorfield>( this->noi, vectorfield( this->nos, Vector3::Zero() ) ); // [noi][nos]
    this->velocities_previous = velocities;                                                              // [noi][nos]
    this->forces_previous     = velocities;                                                              // [noi][nos]
    this->grad                = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->grad_pr             = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->projection          = std::vector<scalar>( this->noi, 0 ); // [noi]
    this->force_norm2         = std::vector<scalar>( this->noi, 0 ); // [noi]
    this->searchdir           = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
};

/*
    Template instantiation of the Simulation class for use with the VP Solver.
        The velocity projection method is often efficient for direct minimization,
        but deals poorly with quickly varying fields or stochastic noise.
    Paper: P. F. Bessarab et al., Method for finding mechanism and activation energy
           of magnetic transitions, applied to skyrmion and antivortex annihilation,
           Comp. Phys. Comm. 196, 335 (2015).

    Instead of the cartesian update scheme with re-normalization, this implementation uses the orthogonal spin
   optimization scheme, described by A. Ivanov in https://arxiv.org/abs/1904.02669.
*/

template<>
inline void Method_Solver<Solver::VP_OSO>::Iteration()
{
    scalar projection_full  = 0;
    scalar force_norm2_full = 0;

    // Set previous
    for( int img = 0; img < noi; ++img )
    {
        auto g    = grad[img].data();
        auto g_pr = grad_pr[img].data();
        auto v    = velocities[img].data();
        auto v_pr = velocities_previous[img].data();

        Backend::par::apply(
            nos,
            [g, g_pr, v, v_pr] SPIRIT_LAMBDA( int idx )
            {
                g_pr[idx] = g[idx];
                v_pr[idx] = v[idx];
            } );
    }

    // Get the forces on the configurations
    this->Calculate_Force( configurations, forces );
    this->Calculate_Force_Virtual( configurations, forces, forces_virtual );

    for( int img = 0; img < this->noi; img++ )
    {
        auto & image = *this->configurations[img];
        auto & grad  = this->grad[img];
        Solver_Kernels::oso_calc_gradients( grad, image, this->forces[img] );
        Vectormath::scale( grad, -1.0 );
    }

    for( int img = 0; img < noi; ++img )
    {
        auto & velocity = velocities[img];
        auto g          = this->grad[img].data();
        auto g_pr       = this->grad_pr[img].data();
        auto v          = velocities[img].data();
        auto m_temp     = this->m;

        // Calculate the new velocity
        Backend::par::apply(
            nos, [g, g_pr, v, m_temp] SPIRIT_LAMBDA( int idx ) { v[idx] += 0.5 / m_temp * ( g_pr[idx] + g[idx] ); } );

        // Get the projection of the velocity on the force
        projection[img]  = Vectormath::dot( velocity, this->grad[img] );
        force_norm2[img] = Vectormath::dot( this->grad[img], this->grad[img] );
    }
    for( int img = 0; img < noi; ++img )
    {
        projection_full += projection[img];
        force_norm2_full += force_norm2[img];
    }
    for( int img = 0; img < noi; ++img )
    {
        auto sd     = this->searchdir[img].data();
        auto v      = this->velocities[img].data();
        auto g      = this->grad[img].data();
        auto m_temp = this->m;

        scalar dt    = this->systems[img]->llg_parameters->dt;
        scalar ratio = projection_full / force_norm2_full;

        // Calculate the projected velocity
        if( projection_full <= 0 )
        {
            Vectormath::fill( velocities[img], { 0, 0, 0 } );
        }
        else
        {
            Backend::par::apply( nos, [g, v, ratio] SPIRIT_LAMBDA( int idx ) { v[idx] = g[idx] * ratio; } );
        }

        Backend::par::apply(
            nos,
            [sd, dt, m_temp, v, g] SPIRIT_LAMBDA( int idx ) { sd[idx] = dt * v[idx] + 0.5 / m_temp * dt * g[idx]; } );
    }
    Solver_Kernels::oso_rotate( this->configurations, this->searchdir );
}

template<>
inline std::string Method_Solver<Solver::VP_OSO>::SolverName()
{
    return "VP_OSO";
};

template<>
inline std::string Method_Solver<Solver::VP_OSO>::SolverFullName()
{
    return "Velocity Projection using exponential transforms";
};