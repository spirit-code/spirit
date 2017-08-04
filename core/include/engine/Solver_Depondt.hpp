template <> inline
void Method_Template<Solver::Depondt>::Solver_Initialise ()
{
    this->forces         = std::vector<vectorfield>( this->noi, vectorfield( this->nos, {0, 0, 0} ) );
    this->forces_virtual = std::vector<vectorfield>( this->noi, vectorfield( this->nos, {0, 0, 0} ) );

    this->rotationaxis = std::vector<vectorfield>( this->noi, vectorfield( this->nos, {0, 0, 0} ) );
    this->forces_virtual_predictor = std::vector<vectorfield>( this->noi, vectorfield( this->nos, {0, 0, 0} ) );
    this->angle = scalarfield( this->nos, 0 );
    this->forces_virtual_norm = std::vector<scalarfield>( this->noi, scalarfield( this->nos, 0 ) );
    
    this->configurations_predictor = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for (int i=0; i<this->noi; i++)
        configurations_predictor[i] = std::shared_ptr<vectorfield>( new vectorfield( this->nos, {0, 0, 0} ) );
    
    this->configurations_temp  = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for (int i=0; i<this->noi; i++)
        configurations_temp[i] = std::shared_ptr<vectorfield>( new vectorfield( this->nos, {0, 0, 0} ) );
        
};


/*
    Template instantiation of the Simulation class for use with the Depondt Solver.
    The Depondt method is an improvement of Heun's method for spin systems. It applies
    rotations instead of finite displacements and thus avoids re-normalizations.
    TODO: reference paper
*/
template <> inline
void Method_Template<Solver::Depondt>::Solver_Iteration ()
{
    // Get the actual forces on the configurations
    this->Calculate_Force( this->configurations, this->forces );
    
    // Optimization for each image
    for (int i = 0; i < this->noi; ++i)
    {
        auto& system         = this->systems[i];
        auto& conf           = *this->configurations[i];
        auto& conf_predictor = *this->configurations_predictor[i];

        scalar dtg = system->llg_parameters->dt * Utility::Constants::gamma / Utility::Constants::mu_B / 
                     ( 1 + pow( system->llg_parameters->damping, 2 )  );
        
        // Calculate Virtual force H
        this->VirtualForce( conf, *system->llg_parameters, forces[i], forces_virtual[i] );
        
        // For Rotation matrix R := R( H_normed, angle )
        Vectormath::norm( forces_virtual[i], angle );   // angle = |forces_virtual|

        Vectormath::set_c_a( 1, forces_virtual[i], rotationaxis[i] );  // rotationaxis = |forces_virtual|
        Vectormath::normalize_vectors( rotationaxis[i] );            // normalize rotation axis 
        
        Vectormath::scale( angle, -dtg );    // angle = |forces_virtual| * dt
        
        // Get spin predictor n' = R(H) * n
        Vectormath::rotate( conf, rotationaxis[i], angle, conf_predictor );  
    }
    
    this->Calculate_Force( configurations_predictor, this->forces );
    
    for (int i=0; i < this->noi; i++)
    {
        auto& system = this->systems[i];
        auto& conf   = *this->configurations[i];

        scalar dtg = system->llg_parameters->dt * Utility::Constants::gamma / Utility::Constants::mu_B / 
                     ( 1 + pow( system->llg_parameters->damping, 2 )  );
        
        // Calculate Predicted Virtual force H'
        this->VirtualForce( *configurations_predictor[i], *system->llg_parameters, forces[i], forces_virtual_predictor[i] );
        
        // Calculate the linear combination of the two forces_virtuals
        Vectormath::scale( forces_virtual[i], 0.5 );   // H = H/2
        Vectormath::add_c_a( 0.5, forces_virtual_predictor[i], forces_virtual[i] ); // H = (H + H')/2
        
        // For Rotation matrix R' := R( H'_normed, angle' )
        Vectormath::norm( forces_virtual[i], angle );   // angle' = |forces_virtual lin combination|
        Vectormath::scale( angle, -dtg );              // angle' = |forces_virtual lin combination| * dt
        
        Vectormath::normalize_vectors( forces_virtual[i] );  // normalize virtual force
        
        // Get new spin conf n_new = R( (H+H')/2 ) * n
        Vectormath::rotate( conf, forces_virtual[i], angle, conf );  
    }
};

template <> inline
std::string Method_Template<Solver::Depondt>::SolverName()
{
	return "Depondt";
};

template <> inline
std::string Method_Template<Solver::Depondt>::SolverFullName()
{
	return "Depondt";
};