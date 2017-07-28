template <> inline
void Method_Template<Solver::NCG>::Solver_Init ()
{
    std::cerr << "NCG INIT" << std::endl;

    this->jmax    = 500;    // max iterations
    this->n       = 50;     // restart every n iterations XXX: what's the appropriate val?
    
    this->tol_NR  = 1e-5;   // Newton-Raphson error tolerance
    this->tol_nCG = 1e-5;   // optimizer's error tolerance1
    this->eps_NR  = pow( this->tol_NR, 2 );   // for Newton-Raphson's convergence condition 
    this->eps_nCG = pow( this->tol_nCG, 2 );  // for optimizer's convergence condition
    
    this->alpha = std::vector<scalarfield>( this->noi, scalarfield( this->nos, 0 ) );
    this->beta  = std::vector<scalarfield>( this->noi, scalarfield( this->nos, 0 ) );
    
    // XXX: right type might be std::vector<scalar> and NOT std::vector<scalarfield>
    this->delta_0   = std::vector<scalarfield>( this->noi, scalarfield( this->nos, 0 ) );
    this->delta_new = std::vector<scalarfield>( this->noi, scalarfield( this->nos, 0 ) );
    this->delta_old = std::vector<scalarfield>( this->noi, scalarfield( this->nos, 0 ) );
    this->delta_d   = std::vector<scalarfield>( this->noi, scalarfield( this->nos, 0 ) );
    
    this->res = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->d   = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    
    this->r_dot_d = std::vector<scalarfield>( this->noi, scalarfield( this->nos, 0 ) );
    this->dda2    = std::vector<scalarfield>( this->noi, scalarfield( this->nos, 0 ) );
    
    this->Calculate_Force( this->configurations, this->force );   // initialize residual
    
    for (int img=0; img<this->noi; img++)
    {
        // set residual = - f'(x)
        Engine::Vectormath::set_c_a( -1, this->force[img], this->res[img] );    
        
        // d = r
        Engine::Vectormath::set_c_a( 1, this->res[img], this->d[img] );         
        
        // delta new = r * r
        Engine::Vectormath::dot( this->res[img], this->res[img], this->delta_new[img] ); 
        
        //save initial delta
        this->delta_0[img] = this->delta_new[img];
    }
};


/*
    Template instantiation of the Simulation class for use with the SIB Solver
*/
template <> inline
void Method_Template<Solver::NCG>::Solver_Step ()
{
    std::cerr << "NCG STEP" << std::endl;

    this->continue_NR = true;   // by default continue Newton-Raphson
    this->restart_nCG = false;  // by default do not restart( XXX:reset?? ) the whole methode
    
    this->eps_NR = std::pow( tol_NR, 2 );   // calculate NR convergence criterion
    
    // calculate delta_d
    for (int img=0; img<this->noi; img++)
        Engine::Vectormath::dot( this->d[img], this->d[img], this->delta_d[img] );  
    
    // Perform a Newton-Raphson line search in order to determine the minimum along d  
    for( int j=0; j<jmax && continue_NR; j++ )
    {
        
        // alpha = - (f'*d)/(d*f''*d)	// TODO: How to get the second derivative from here??
        //this->alpha = - 
        //this->method->Calculate( )
        //this->alpha = - Engine::Vectormath::dot( )
        
        // x = x + alpha*d where x is the configurations
        for (int img=0; img<this->noi; img++ )
            Engine::Vectormath::add_c_a( this->alpha[img], this->d[img], *this->configurations[img] );   
        
        // check convergence
        
        for ( int img=0; img<this->noi && this->continue_NR == true; img++)
        {
            // alpha squareing. alpha[i] = alpha[i]*alpha[i]. In next iteration we will have new alpha
            Engine::Vectormath::dot( this->alpha[img], this->alpha[img], this->alpha[img] ); 
            
            // calculate dda2[i] = delta_d[i] * alpha_sq[i]
            Engine::Vectormath::dot( this->delta_d[img], this->alpha[img], this->dda2[img] );
            
            //  TODO: replace with minmax           
            // check for every spin of that image 
            for ( int sp=0; sp<this->nos; sp++ )
            {
                if( this->dda2[img][sp] > this->eps_NR )
                {
                    this->continue_NR = false;    // stop Newton-Raphson
                    break;                        // break the loop over image
                    // the outermost loop will not continue due to its conditional and NR will stop 
                }
            }
        } // end of convergence test
    
    } // end Newton-Raphson
    
    // calculate force from the new configurations
    this->Calculate_Force( this->configurations, this->force );
    
    for (int img=0; img<this->noi; img++)
    {
        // set residual = - f'(x)
        Engine::Vectormath::set_c_a( -1, this->force[img], this->res[img] ); 
        
        // delta_old = delta_new
        this->delta_old[img] = this->delta_new[img];
        
        // delta_new = r * r
        Engine::Vectormath::dot( this->res[img], this->res[img], this->delta_new[img] );  
        
        // beta = delta_new / delta_old
        Engine::Vectormath::divide( this->delta_new[img], this->delta_old[img], this->beta[img] );
        
        // d = r + beta*d
        Engine::Vectormath::set_c_a( this->beta[img], this->d[img], this->d[img] );   // d = beta*d
        Engine::Vectormath::add_c_a( 1, this->res[img], this->d[img] );               // d += r
    }
        
    // Restart if d is not a descent direction or after nos iterations
    //    The latter improves convergence for small nos
    
    // XXX: probably the issuing of the restarting can be in a seperate function since it has
    // its own semantics
    
    // in case you are in the nth iteration restart nCG
    if ( this->iteration > 0 && ( this->iteration % this->n ) != 0 )
    this->restart_nCG = true;

    // in case there is no previous request to restart nCG check restarting criterion rd < 0 
    if ( this->restart_nCG == false )
    {
        this->restart_nCG = true; // set restarting to true
    
        for (int img=0; img<this->noi && this->restart_nCG == true; img++)
        {
            // r_dod_d = res * d for that image
            Engine::Vectormath::dot( this->res[img], this->d[img], this->r_dot_d[img] );
            
            // TODO: better check with minmax_component

            for (int sp=0; sp<this->nos; sp++)
            {
                // in case of r*d[img][sp] not meeting the restarting criterion set restarting to false
                // break this loop and the outermost loop will not continue due to its conditional
                if( this->r_dot_d[img][sp] > 0 )
                {
                    this->restart_nCG = false;
                    break;    
                }
            }
        }
    }
    
    // reset d in case of restart
    if ( this->restart_nCG )
    {
        for (int img=0; img<this->noi; img++)
            Engine::Vectormath::set_c_a( 1, { 0, 0, 0 }, this->d[img] );
    }
};