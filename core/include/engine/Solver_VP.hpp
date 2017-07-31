template <> inline
void Method_Template<Solver::VP>::Solver_Initialise ()
{
    this->spins_temp  = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for (int i=0; i<this->noi; i++)
      spins_temp[i] = std::shared_ptr<vectorfield>(new vectorfield(this->nos));

    this->velocity = std::vector<vectorfield>(this->noi, vectorfield(this->nos, Vector3::Zero()));	// [noi][nos]
    this->velocity_previous = velocity;	// [noi][nos]
    this->force_previous = velocity;	// [noi][nos]
    this->projection = std::vector<scalar>(this->noi, 0);	// [noi]
    this->force_norm2 = std::vector<scalar>(this->noi, 0);	// [noi]
};


/*
    Template instantiation of the Simulation class for use with the VP Solver.
    The velocity projection method is often efficient for direct minimization,
    but deals poorly with quickly varying fields or stochastic noise.
    TODO: reference paper
*/
template <> inline
void Method_Template<Solver::VP>::Solver_Iteration ()
{
    std::shared_ptr<Data::Spin_System> s;
    scalar projection_full  = 0;
    scalar force_norm2_full = 0;

    // Set previous
    for (int i = 0; i < noi; ++i)
    {
        Vectormath::set_c_a(1.0, force[i], force_previous[i]);
        Vectormath::set_c_a(1.0, velocity[i], velocity_previous[i]);
    }

    // Get the forces on the configurations
    this->Calculate_Force(configurations, force);

    for (int i = 0; i < noi; ++i)
    {
        auto& l_velocity = velocity[i];
        auto& l_force = force[i];
        auto& l_force_prev = force_previous[i];
        auto& configuration = *(configurations[i]);

        s = this->systems[i];
        scalar dt = s->llg_parameters->dt;

        // Calculate the new velocity
        Vectormath::add_c_a(0.5 / m * dt, l_force_prev, l_velocity);
        Vectormath::add_c_a(0.5 / m * dt, l_force, l_velocity);

        // Get the projection of the velocity on the force
        projection[i] = Vectormath::dot(l_velocity, l_force);
        force_norm2[i] = Vectormath::dot(l_force, l_force);
    }
    for (int i = 0; i < noi; ++i)
    {
        projection_full += projection[i];
        force_norm2_full += force_norm2[i];
    }
    for (int i = 0; i < noi; ++i)
    {
        auto& l_velocity = velocity[i];
        auto& l_force = force[i];
        auto& l_force_prev = force_previous[i];
        auto& configuration = *(configurations[i]);

        s = this->systems[i];
        scalar dt = s->llg_parameters->dt;

        // Calculate the projected velocity
        if (projection_full <= 0)
        {
            Vectormath::fill(l_velocity, { 0,0,0 });
        }
        else
        {
            Vectormath::set_c_a(1.0, l_force, l_velocity);
            Vectormath::scale(l_velocity, projection_full / force_norm2_full);
        }

        // Copy in
        Vectormath::set_c_a(1.0, configuration, *spins_temp[i]);

        // Move the spins
        Vectormath::add_c_a(dt, l_velocity, *spins_temp[i]);
        Vectormath::add_c_a(0.5/m*dt*dt, l_force, *spins_temp[i]);
        Vectormath::normalize_vectors(*spins_temp[i]);

        // Copy out
        Vectormath::set_c_a(1.0, *spins_temp[i], configuration);
    }
};

template <> inline
std::string Method_Template<Solver::VP>::SolverName()
{
	return "VP";
};

template <> inline
std::string Method_Template<Solver::VP>::SolverFullName()
{
	return "Velocity Projection";
};