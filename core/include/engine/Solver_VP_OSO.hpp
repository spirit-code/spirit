template <> inline
void Method_Solver<Solver::VP_OSO>::Initialize ()
{
    this->forces         = std::vector<vectorfield>( this->noi, vectorfield( this->nos, {0, 0, 0} ) );
    this->forces_virtual = std::vector<vectorfield>( this->noi, vectorfield( this->nos, {0, 0, 0} ) );

    this->configurations_temp  = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for (int i=0; i<this->noi; i++)
      configurations_temp[i] = std::shared_ptr<vectorfield>(new vectorfield(this->nos));

    this->velocities          = std::vector<vectorfield>(this->noi, vectorfield(this->nos, Vector3::Zero()));	// [noi][nos]
    this->velocities_previous = velocities;	// [noi][nos]
    this->forces_previous     = velocities;	// [noi][nos]
    this->grad    = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0,0,0 } ) );
    this->grad_pr = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0,0,0 } ) );
    this->projection          = std::vector<scalar>(this->noi, 0);	// [noi]
    this->force_norm2         = std::vector<scalar>(this->noi, 0);	// [noi]

    this->searchdir = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0,0,0 } ) );

};


/*
    Template instantiation of the Simulation class for use with the VP Solver.
		The velocity projection method is often efficient for direct minimization,
		but deals poorly with quickly varying fields or stochastic noise.
	Paper: P. F. Bessarab et al., Method for finding mechanism and activation energy
		   of magnetic transitions, applied to skyrmion and antivortex annihilation,
		   Comp. Phys. Comm. 196, 335 (2015).
*/
template <> inline
void Method_Solver<Solver::VP_OSO>::Iteration ()
{
    scalar projection_full  = 0;
    scalar force_norm2_full = 0;

    // Set previous
    for (int img = 0; img < noi; ++img)
    {
        auto& image = *this->configurations[img];
        for (int i = 0; i < this->nos; ++i)
        {
            this->forces_virtual[img][i] = image[i].cross(this->forces[img][i]);
        }
        Vectormath::set_c_a(1.0, forces[img],   forces_previous[img]);
        Vectormath::set_c_a(1.0, velocities[img], velocities_previous[img]);
    }

    // Get the forces on the configurations
    this->Calculate_Force(configurations, forces);
    this->Calculate_Force_Virtual(configurations, forces, forces_virtual);


    #pragma omp parallel for
    for( int img=0; img < this->noi; img++ )
    {
        auto& image = *this->configurations[img];
        auto& grad_ref = this->grad[img];
        for (int i = 0; i < this->nos; ++i){
            this->forces_virtual[img][i] = image[i].cross(this->forces[img][i]);
        }
        Solver_Kernels::oso_calc_gradients(grad_ref, image, this->forces[img]);
    }

    for (int i = 0; i < noi; ++i)
    {
        auto& velocity      = velocities[i];
        auto& force         = this->grad[i];
        auto& force_prev    = this->grad_pr[i];

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
        auto& force              = this->grad[i];
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
        Vectormath::set_c_a(-dt, velocity, this->searchdir[i]);
        Vectormath::add_c_a(-0.5 / m * dt, force, this->searchdir[i]); // Note: as force is scaled with dt, this corresponds to dt^2
    }
    Solver_Kernels::oso_rotate( this->configurations, this->searchdir);
}

template <> inline
std::string Method_Solver<Solver::VP_OSO>::SolverName()
{
	return "VP_OSO";
};

template <> inline
std::string Method_Solver<Solver::VP_OSO>::SolverFullName()
{
	return "Velocity Projection";
};