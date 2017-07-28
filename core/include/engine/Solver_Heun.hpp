template <> inline
void Method_Template<Solver::Heun>::Solver_Init ()
{
    std::cerr << "Heun INIT" << std::endl;

    this->virtualforce = std::vector<vectorfield>( this->noi, vectorfield( this->nos, {0, 0, 0} ) );
    
    this->spins_temp  = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for (int i=0; i<this->noi; i++)
      spins_temp[i] = std::shared_ptr<vectorfield>(new vectorfield(this->nos));
  
    this->spins_predictor = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for (int i=0; i<this->noi; i++)
      spins_predictor[i] = std::shared_ptr<vectorfield>(new vectorfield(this->nos));  
    
    this->temp2 = vectorfield( this->nos, {0, 0, 0} );
    this->temp1 = vectorfield( this->nos, {0, 0, 0} );
};


/*
    Template instantiation of the Simulation class for use with the SIB Solver
*/
template <> inline
void Method_Template<Solver::Heun>::Solver_Step ()
{
    std::cerr << "Heun STEP" << std::endl;

    // Get the actual forces on the configurations
    this->Calculate_Force( this->configurations, this->force );
    
    // Optimization for each image
    for (int i = 0; i < this->noi; ++i)
    {
        this->s = systems[i];
        auto& conf = *this->configurations[i];
        
        // First step - Predictor
        this->VirtualForce( *s->spins, *s->llg_parameters, force[i], xi, virtualforce[i] );
        
        Vectormath::set_c_cross( -1, conf, virtualforce[i], *spins_temp[i] );  // temp1 = -( conf x A )
        Vectormath::set_c_a( 1, conf, *spins_predictor[i] );                   // spins_predictor = conf
        Vectormath::add_c_a( 1, *spins_temp[i], *spins_predictor[i] );         // spins_predictor = conf + dt*temp1
        
        // Normalize spins
        Vectormath::normalize_vectors( *spins_predictor[i] );
    }
    
    // Calculate_Force for the Corrector
    this->Calculate_Force( spins_predictor, this->force );
    
    for (int i=0; i < this->noi; i++)
    {
        this->s = systems[i];
        auto& conf = *this->configurations[i];

        // Second step - Corrector
        this->VirtualForce( *spins_predictor[i], *s->llg_parameters, force[i], xi, virtualforce[i] );
        
        Vectormath::scale( *spins_temp[i], 0.5 );                                     // spins_temp = 0.5 * spins_temp
        Vectormath::add_c_a( 1, conf, *spins_temp[i] );                               // spins_temp = conf + 0.5 * spins_temp 
        Vectormath::set_c_cross( -1, *spins_predictor[i], virtualforce[i], temp1 );   // temp1 = - ( conf' x A' )
        Vectormath::add_c_a( 0.5, temp1, *spins_temp[i] );                            // spins_temp = conf + 0.5 * spins_temp + 0.5 * temp1        

        // Normalize spins
        Vectormath::normalize_vectors( *spins_temp[i] );
        
        // Copy out
        conf = *spins_temp[i];
    } 
};