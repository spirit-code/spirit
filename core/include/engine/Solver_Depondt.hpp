template <> inline
void Method_Solver<Solver::Depondt>::Initialize ()
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
void Method_Solver<Solver::Depondt>::Iteration ()
{
    // Get the actual forces on the configurations
    this->Calculate_Force_Virtual( this->configurations, this->forces_virtual );
    
    // Optimization for each image
    for (int i = 0; i < this->noi; ++i)
    {
        auto& conf           = *this->configurations[i];
        auto& conf_predictor = *this->configurations_predictor[i];

        // For Rotation matrix R := R( H_normed, angle )
        Vectormath::norm( forces_virtual[i], angle );   // angle = |forces_virtual|

        Vectormath::set_c_a( 1, forces_virtual[i], rotationaxis[i] );  // rotationaxis = |forces_virtual|
        Vectormath::normalize_vectors( rotationaxis[i] );            // normalize rotation axis 
        
        // Get spin predictor n' = R(H) * n
        Vectormath::rotate( conf, rotationaxis[i], angle, conf_predictor );  
    }
    
    this->Calculate_Force_Virtual( configurations_predictor, this->forces_virtual_predictor );
    
    for (int i=0; i < this->noi; i++)
    {
        auto& conf   = *this->configurations[i];

        // Calculate the linear combination of the two forces_virtuals
        Vectormath::scale( forces_virtual[i], 0.5 );   // H = H/2
        Vectormath::add_c_a( 0.5, forces_virtual_predictor[i], forces_virtual[i] ); // H = (H + H')/2
        
        // For Rotation matrix R' := R( H'_normed, angle' )
        Vectormath::norm( forces_virtual[i], angle );   // angle' = |forces_virtual lin combination|
        
        Vectormath::normalize_vectors( forces_virtual[i] );  // normalize virtual force
        
        // Get new spin conf n_new = R( (H+H')/2 ) * n
        Vectormath::rotate( conf, forces_virtual[i], angle, conf );  
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