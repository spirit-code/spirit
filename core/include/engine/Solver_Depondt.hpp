template <> inline
void Method_Solver<Solver::Depondt>::Initialize ()
{
    this->forces         = std::vector<vectorfield>( this->noi, vectorfield( this->nos, {0, 0, 0} ) );
    this->forces_virtual = std::vector<vectorfield>( this->noi, vectorfield( this->nos, {0, 0, 0} ) );

    this->forces_predictor = std::vector<vectorfield>( this->noi, vectorfield( this->nos, {0, 0, 0} ) );
    this->forces_virtual_predictor = std::vector<vectorfield>( this->noi, vectorfield( this->nos, {0, 0, 0} ) );

    this->rotationaxis = std::vector<vectorfield>( this->noi, vectorfield( this->nos, {0, 0, 0} ) );
    this->angle = scalarfield( this->nos, 0 );
    this->forces_virtual_norm = std::vector<scalarfield>( this->noi, scalarfield( this->nos, 0 ) );

    this->configurations_predictor = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for (int i=0; i<this->noi; i++)
        configurations_predictor[i] = std::shared_ptr<vectorfield>( new vectorfield( this->nos, {0, 0, 0} ) );

    this->temp1 = vectorfield( this->nos, {0, 0, 0} );
};


/*
    Template instantiation of the Simulation class for use with the Depondt Solver.
        The Depondt method is an improvement of Heun's method for spin systems. It applies
        rotations instead of finite displacements and thus avoids re-normalizations.
    Paper: Ph. Depondt et al., Spin dynamics simulations of two-dimensional clusters with
           Heisenberg and dipole-dipole interactions, J. Phys. Condens. Matter 21, 336005 (2009).
*/
template <> inline
void Method_Solver<Solver::Depondt>::Iteration ()
{
    // Generate random vectors for this iteration
    this->Prepare_Thermal_Field();

    // Get the actual forces on the configurations
    this->Calculate_Force(this->configurations, this->forces);
    this->Calculate_Force_Virtual(this->configurations, this->forces, this->forces_virtual);

    // Predictor for each image
    for (int i = 0; i < this->noi; ++i)
    {
        auto& conf           = *this->configurations[i];
        auto& conf_predictor = *this->configurations_predictor[i];
        auto anglep = angle.data();
        auto axisp = rotationaxis[i].data();
        auto f_virtual = forces_virtual[i].data();

        // For Rotation matrix R := R( H_normed, angle )
        Backend::par::apply(nos, [anglep, axisp, f_virtual] SPIRIT_LAMBDA (int idx) {
            anglep[idx] = f_virtual[idx].norm(); // angle = |forces_virtual|
            axisp[idx] = f_virtual[idx].normalized(); // rotationaxis = forces_virtual/|forces_virtual|
        });

        // Get spin predictor n' = R(H) * n
        Vectormath::rotate( conf, rotationaxis[i], angle, conf_predictor );
    }

    // Calculate_Force for the Corrector
    this->Calculate_Force(this->configurations_predictor, this->forces_predictor);
    this->Calculate_Force_Virtual(this->configurations_predictor, this->forces_predictor, this->forces_virtual_predictor);

    // Corrector step for each image
    for (int i=0; i < this->noi; i++)
    {
        auto& conf   = *this->configurations[i];
        auto temp1p = temp1.data();
        auto anglep = angle.data();
        auto f_virtual = forces_virtual[i].data();
        auto f_virtual_predictor = forces_virtual_predictor[i].data();

        Backend::par::apply(nos, [temp1p, anglep, f_virtual, f_virtual_predictor] SPIRIT_LAMBDA (int idx) {
            // Calculate the linear combination of the two forces_virtuals
            temp1p[idx] = 0.5 * (f_virtual[idx] + f_virtual_predictor[idx]);
            // Get the rotation angle as norm of temp1 ...For Rotation matrix R' := R( H'_normed, angle' )
            anglep[idx] = temp1p[idx].norm();
            // Normalize temp1 to get rotation axes
            temp1p[idx].normalize();
        });

        // Get new spin conf n_new = R( (H+H')/2 ) * n
        Vectormath::rotate( conf, temp1, angle, conf );
    }
};

template <> inline
std::string Method_Solver<Solver::Depondt>::SolverName()
{
    return "Depondt";
};

template <> inline
std::string Method_Solver<Solver::Depondt>::SolverFullName()
{
    return "Depondt";
};