template <> inline
void Method_Solver<Solver::VP>::Initialize ()
{
    this->forces         = std::vector<vectorfield>( this->noi, vectorfield( this->nos, {0, 0, 0} ) );
    this->forces_virtual = std::vector<vectorfield>( this->noi, vectorfield( this->nos, {0, 0, 0} ) );

    this->configurations_temp  = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for (int i=0; i<this->noi; i++)
      configurations_temp[i] = std::shared_ptr<vectorfield>(new vectorfield(this->nos));

    this->velocities          = std::vector<vectorfield>(this->noi, vectorfield(this->nos, Vector3::Zero()));	// [noi][nos]
    this->velocities_previous = velocities;	// [noi][nos]
    this->forces_previous     = velocities;	// [noi][nos]
    this->projection          = std::vector<scalar>(this->noi, 0);	// [noi]
    this->force_norm2         = std::vector<scalar>(this->noi, 0);	// [noi]
};


/*
    Template instantiation of the Simulation class for use with the VP Solver.
    The velocity projection method is often efficient for direct minimization,
    but deals poorly with quickly varying fields or stochastic noise.
    TODO: reference paper
*/
template <> inline
void Method_Solver<Solver::VP>::Iteration ()
{
    scalar projection_full  = 0;
    scalar force_norm2_full = 0;

    // Set previous
    for (int i = 0; i < noi; ++i)
    {
        Vectormath::set_c_a(1.0, forces[i],   forces_previous[i]);
        Vectormath::set_c_a(1.0, velocities[i], velocities_previous[i]);
    }

    // Get the forces on the configurations
    this->Calculate_Force_Virtual(configurations, forces);
    
    for (int i = 0; i < noi; ++i)
    {
        auto& velocity      = velocities[i];
        auto& force         = forces[i];
        auto& force_prev    = forces_previous[i];

        // Calculate the new velocity
        Vectormath::add_c_a(0.5/m, force_prev, velocity);
        Vectormath::add_c_a(0.5/m, force, velocity);

        // Get the projection of the velocity on the force
        projection[i] = Vectormath::dot(velocity, force);
        force_norm2[i] = Vectormath::dot(force, force);
    }
    for (int i = 0; i < noi; ++i)
    {
        projection_full += projection[i];
        force_norm2_full += force_norm2[i];
    }
    for (int i = 0; i < noi; ++i)
    {
        auto& velocity           = velocities[i];
        auto& force              = forces[i];
        auto& configuration      = *(configurations[i]);
        auto& configuration_temp = *(configurations_temp[i]);

        scalar dt = this->systems[i]->llg_parameters->dt;

        // Calculate the projected velocity
        if (projection_full <= 0)
        {
            Vectormath::fill(velocity, { 0,0,0 });
        }
        else
        {
            Vectormath::set_c_a(1.0, force, velocity);
            Vectormath::scale(velocity, projection_full / force_norm2_full);
        }

        // Copy in
        Vectormath::set_c_a(1.0, configuration, configuration_temp);

        // Move the spins
        Vectormath::add_c_a(dt, velocity, configuration_temp);
        Vectormath::add_c_a(0.5 / m * dt, force, configuration_temp); // Note: as force is scaled with dt, this corresponds to dt^2
        Vectormath::normalize_vectors(configuration_temp);

        // Copy out
        Vectormath::set_c_a(1.0, configuration_temp, configuration);
    }
};

template <> inline
std::string Method_Solver<Solver::VP>::SolverName()
{
	return "VP";
};

template <> inline
std::string Method_Solver<Solver::VP>::SolverFullName()
{
	return "Velocity Projection";
};