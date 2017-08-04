template <> inline
void Method_Template<Solver::Heun>::Solver_Initialise ()
{
    this->forces         = std::vector<vectorfield>( this->noi, vectorfield( this->nos, {0, 0, 0} ) );
    this->forces_virtual = std::vector<vectorfield>( this->noi, vectorfield( this->nos, {0, 0, 0} ) );
    
    this->configurations_temp  = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for (int i=0; i<this->noi; i++)
      configurations_temp[i] = std::shared_ptr<vectorfield>(new vectorfield(this->nos));
  
    this->configurations_predictor = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for (int i=0; i<this->noi; i++)
      configurations_predictor[i] = std::shared_ptr<vectorfield>(new vectorfield(this->nos));  
    
    this->temp1 = vectorfield( this->nos, {0, 0, 0} );
};


/*
    Template instantiation of the Simulation class for use with the Heun Solver.
    The Heun method is a basic solver for the PDE at hand here. It is sufficiently
    efficient and stable.
    TODO: reference paper
*/
template <> inline
void Method_Template<Solver::Heun>::Solver_Iteration ()
{
    // Get the actual forces on the configurations
    this->Calculate_Force( this->configurations, this->forces );
    
    // Optimization for each image
    for (int i = 0; i < this->noi; ++i)
    {
        auto& system         = this->systems[i];
        auto& conf           = *this->configurations[i];
        auto& conf_temp      = *this->configurations_temp[i];
        auto& conf_predictor = *this->configurations_predictor[i];
        
        // First step - Predictor
        this->VirtualForce( conf, *system->llg_parameters, forces[i], forces_virtual[i] );
        
        Vectormath::set_c_cross( 1, conf, forces_virtual[i], conf_temp );  // temp1 = -( conf x A )
        Vectormath::set_c_a( 1, conf, conf_predictor );                   // configurations_predictor = conf
        Vectormath::add_c_a( 1, conf_temp, conf_predictor );         // configurations_predictor = conf + dt*temp1
        
        // Normalize spins
        Vectormath::normalize_vectors( conf_predictor );
    }
    
    // Calculate_Force for the Corrector
    this->Calculate_Force( configurations_predictor, this->forces );
    
    for (int i=0; i < this->noi; i++)
    {
        auto& system         = this->systems[i];
        auto& conf           = *this->configurations[i];
        auto& conf_temp      = *this->configurations_temp[i];
        auto& conf_predictor = *this->configurations_predictor[i];

        // Second step - Corrector
        this->VirtualForce( conf_predictor, *system->llg_parameters, forces[i], forces_virtual[i] );
        
        Vectormath::scale( conf_temp, 0.5 );                                     // configurations_temp = 0.5 * configurations_temp
        Vectormath::add_c_a( 1, conf, conf_temp );                               // configurations_temp = conf + 0.5 * configurations_temp 
        Vectormath::set_c_cross( 1, conf_predictor, forces_virtual[i], temp1 );   // temp1 = - ( conf' x A' )
        Vectormath::add_c_a( 0.5, temp1, conf_temp );                            // configurations_temp = conf + 0.5 * configurations_temp + 0.5 * temp1        

        // Normalize spins
        Vectormath::normalize_vectors( conf_temp );
        
        // Copy out
        conf = conf_temp;
    } 
};

template <> inline
std::string Method_Template<Solver::Heun>::SolverName()
{
	return "Heun";
};

template <> inline
std::string Method_Template<Solver::Heun>::SolverFullName()
{
	return "Heun";
};