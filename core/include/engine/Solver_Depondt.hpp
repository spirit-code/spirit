template <> inline
void Method_Template<Solver::Depondt>::Solver_Initialise ()
{
    this->virtualforce = std::vector<vectorfield>( this->noi, vectorfield( this->nos, {0, 0, 0} ) );
    this->rotationaxis = std::vector<vectorfield>( this->noi, vectorfield( this->nos, {0, 0, 0} ) );
    this->virtualforce_predictor = std::vector<vectorfield>( this->noi, vectorfield( this->nos, {0, 0, 0} ) );
    
    this->spins_predictor = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for (int i=0; i<this->noi; i++)
        spins_predictor[i] = std::shared_ptr<vectorfield>( new vectorfield( this->nos, {0, 0, 0} ) );
    
    this->spins_temp  = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for (int i=0; i<this->noi; i++)
        spins_temp[i] = std::shared_ptr<vectorfield>( new vectorfield( this->nos, {0, 0, 0} ) );
        
    this->angle = scalarfield( this->nos, 0 );
    this->virtualforce_norm = std::vector<scalarfield>( this->noi, scalarfield( this->nos, 0 ) );
};


/*
    Template instantiation of the Simulation class for use with the Depondt Solver.
    The Depondt method is an improvement of Heun's method for spin systems. It applies
    rotations instead of finite displacements and thus avoids re-normalizations.
*/
template <> inline
void Method_Template<Solver::Depondt>::Solver_Iteration ()
{
    // Get the actual forces on the configurations
    this->Calculate_Force( this->configurations, this->force );
    
    // Optimization for each image
    for (int i = 0; i < this->noi; ++i)
    {
        this->s = this->systems[i];
        auto& conf = *this->configurations[i];
        this->dtg = this->s->llg_parameters->dt * Utility::Constants::gamma / Utility::Constants::mu_B / 
                    ( 1 + pow( this->s->llg_parameters->damping, 2 )  );
        
        // Calculate Virtual force H
        this->VirtualForce( *s->spins, *s->llg_parameters, force[i], xi, virtualforce[i] );
        
        // For Rotation matrix R := R( H_normed, angle )
        Vectormath::norm( virtualforce[i], angle );   // angle = |virtualforce|

        Vectormath::set_c_a( 1, virtualforce[i], rotationaxis[i] );  // rotationaxis = |virtualforce|
        Vectormath::normalize_vectors( rotationaxis[i] );            // normalize rotation axis 
        
        Vectormath::scale( angle, -dtg );    // angle = |virtualforce| * dt
        
        // Get spin predictor n' = R(H) * n
        Vectormath::rotate( conf, rotationaxis[i], angle, *spins_predictor[i] );  
    }
    
    this->Calculate_Force( spins_predictor, this->force );
    
    for (int i=0; i < this->noi; i++)
    {
        this->s = this->systems[i];
        auto& conf = *this->configurations[i];
        this->dtg = this->s->llg_parameters->dt * Utility::Constants::gamma / Utility::Constants::mu_B / 
                    ( 1 + pow( this->s->llg_parameters->damping, 2 )  );
        
        // Calculate Predicted Virtual force H'
        this->VirtualForce( *spins_predictor[i], *s->llg_parameters, force[i], xi, virtualforce_predictor[i] );
        
        // Calculate the linear combination of the two virtualforces
        Vectormath::scale( virtualforce[i], 0.5 );   // H = H/2
        Vectormath::add_c_a( 0.5, virtualforce_predictor[i], virtualforce[i] ); // H = (H + H')/2
        
        // For Rotation matrix R' := R( H'_normed, angle' )
        Vectormath::norm( virtualforce[i], angle );   // angle' = |virtualforce lin combination|
        Vectormath::scale( angle, -dtg );              // angle' = |virtualforce lin combination| * dt
        
        Vectormath::normalize_vectors( virtualforce[i] );  // normalize virtual force
        
        // Get new spin conf n_new = R( (H+H')/2 ) * n
        Vectormath::rotate( conf, virtualforce[i], angle, conf );  
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