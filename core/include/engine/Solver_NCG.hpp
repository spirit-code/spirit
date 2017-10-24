template <> inline
void Method_Solver<Solver::NCG>::Initialize ()
{
    this->jmax    = 500;    // max iterations
    this->n       = 50;     // restart every n iterations XXX: what's the appropriate val?
    
    this->tolerance_NR  = 1e-5;   // Newton-Raphson error tolerance
    this->tolerance_nCG = 1e-5;   // solver's error tolerance1
	this->epsilon_NR  = 1e-5;// pow(this->tolerance_NR, 2);   // for Newton-Raphson's convergence condition 
    this->epsilon_nCG = pow( this->tolerance_nCG, 2 );  // for solver's convergence condition
    
    this->alpha = std::vector<scalarfield>( this->noi, scalarfield( this->nos, 0 ) );
    this->beta  = std::vector<scalarfield>( this->noi, scalarfield( this->nos, 0 ) );
    
    // XXX: right type might be std::vector<scalar> and NOT std::vector<scalarfield>
    this->delta_0   = std::vector<scalarfield>( this->noi, scalarfield( this->nos, 0 ) );
    this->delta_new = std::vector<scalarfield>( this->noi, scalarfield( this->nos, 0 ) );
    this->delta_old = std::vector<scalarfield>( this->noi, scalarfield( this->nos, 0 ) );
    this->delta_d   = std::vector<scalarfield>( this->noi, scalarfield( this->nos, 0 ) );
    
    this->residual  = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->direction = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    
    this->r_dot_d = std::vector<scalarfield>( this->noi, scalarfield( this->nos, 0 ) );
    this->dda2    = std::vector<scalarfield>( this->noi, scalarfield( this->nos, 0 ) );
    
	// F = - grad
    this->Calculate_Force( this->configurations, this->forces );
    
    for (int img=0; img<this->noi; img++)
    {
        // Project force into the tangent space of the spin configuration
        Manifoldmath::project_tangential(this->forces[img], *this->configurations[img]);

        // set residual = - f'(x)
        Vectormath::set_c_a( 1, this->forces[img], this->residual[img] );
        
        // direction = residual
        Vectormath::set_c_a( 1, this->residual[img], this->direction[img] );
        
        // delta new = r * r
        Vectormath::dot( this->residual[img], this->residual[img], this->delta_new[img] );
        
        // save initial delta
        this->delta_0[img] = this->delta_new[img];
    }
};


/*
    Template instantiation of the Simulation class for use with the NCG Solver
    The method of nonlinear conjugate gradients is a proven and effective solver.
    TODO: reference painless conjugate gradients
*/
template <> inline
void Method_Solver<Solver::NCG>::Iteration ()
{
	// By default continue Newton-Raphson
    this->continue_NR = true;
	// By default do not restart( XXX:reset?? ) the whole method
    this->restart_nCG = false;
    
    // Calculate delta_d
    for (int img=0; img<this->noi; img++)
        Engine::Vectormath::dot( this->direction[img], this->direction[img], this->delta_d[img] );
    
    // Perform a Newton-Raphson line search in order to determine the minimum along d  
    for( int j=0; j<jmax && continue_NR; j++ )
    {
        // Calculate force F = - grad
        this->Calculate_Force(this->configurations, this->forces);

		// Do line search per image
        for (int img = 0; img < this->noi; img++)
        {
            // Project force into the tangent space of the spin configuration
            Manifoldmath::project_tangential(this->forces[img], *this->configurations[img]);

            // Calculate Hessian
            auto hessian = MatrixX(3 * this->nos, 3 * this->nos);
            this->systems[img]->hamiltonian->Hessian(*this->configurations[img], hessian);

            // Calculate alpha (NR step length)
            // alpha = - (f'*d)/(d*f''*d)	// TODO: How to get the second derivative from here??
            Eigen::Ref<VectorX> direction_ref = Eigen::Map<VectorX>(this->direction[img][0].data(), this->nos);
            scalar denominator = direction_ref.dot( hessian * direction_ref );
            scalar numerator = Engine::Vectormath::dot(this->forces[img], this->direction[img]); // / ppp;
            scalar ratio = 1;
            if (std::abs(denominator) > 0)
                ratio = numerator / denominator;
            Vectormath::fill(this->alpha[img], ratio);

            // Update the spins (NR step)
            // x = x + alpha*direction where x is the spin configuration
            // Engine::Vectormath::add_c_a(this->alpha[img], this->direction[img], *this->configurations[img]);
            Engine::Vectormath::add_c_a(ratio, this->direction[img], *this->configurations[img]);
            Vectormath::normalize_vectors(*this->configurations[img]);
                
			// Check NR convergence
            // Square alpha: alpha[i] = alpha[i]*alpha[i] (in next iteration we will have new alpha)
            Engine::Vectormath::dot( this->alpha[img], this->alpha[img], this->alpha[img] ); 
            // Calculate dda2[i] = delta_d[i] * alpha_sq[i]
            Engine::Vectormath::dot( this->delta_d[img], this->alpha[img], this->dda2[img] );
            // Check that convergence is achieved for every spin of that image
            scalar dmax = 0;
            for (auto& x : this->dda2[img]) dmax = std::max(dmax, std::abs(x));
            if (dmax < this->epsilon_NR)
            {
                this->continue_NR = false; // stop Newton-Raphson
                break;                     // break the convergence check
            }
        } // end of convergence test
    } // end Newton-Raphson
    
    // Calculate force F = - grad from the new configurations
    this->Calculate_Force( this->configurations, this->forces );
    
    // Update the direction
    for (int img=0; img<this->noi; img++)
    {
        // Project force into the tangent space of the spin configuration
        Manifoldmath::project_tangential(this->forces[img], *this->configurations[img]);

        // Set residual = - f'(x)
        Engine::Vectormath::set_c_a( 1, this->forces[img], this->residual[img] );

        // delta_old = delta_new
        this->delta_old[img] = this->delta_new[img];

        // delta_new = residual * residual
        Engine::Vectormath::dot( this->residual[img], this->residual[img], this->delta_new[img] );

        // beta = delta_new / delta_old
        Engine::Vectormath::divide( this->delta_new[img], this->delta_old[img], this->beta[img] );

        // direction = residual + beta*direction
        Engine::Vectormath::set_c_a( this->beta[img], this->direction[img], this->direction[img] ); // direction = beta*direction
        Engine::Vectormath::add_c_a( 1, this->residual[img], this->direction[img] );                // direction += residual
    }

    // Restart if direction is not a descent direction or after nos iterations
    //    The latter improves convergence for small nos

    // XXX: probably the issuing of the restarting can be in a seperate function since it has
    // its own semantics

    // In case you are in the nth iteration restart nCG
    if ( this->iteration > 0 && ( this->iteration % this->n ) != 0 )
    this->restart_nCG = true;

    // In case there is no previous request to restart nCG check restarting criterion rd < 0 
    if ( this->restart_nCG == false )
    {
        this->restart_nCG = true; // set restarting to true
    
        for (int img=0; img<this->noi && this->restart_nCG == true; img++)
        {
            // r_dot_d = residual * direction for that image
            Engine::Vectormath::dot( this->residual[img], this->direction[img], this->r_dot_d[img] );
            
            // TODO: better check with minmax_component
            for (int sp=0; sp<this->nos; sp++)
            {
                // In case of residual*direction not meeting the restarting criterion set restarting to false
                // break this loop and the outermost loop will not continue due to its conditional
                if( this->r_dot_d[img][sp] > 0 )
                {
                    this->restart_nCG = false;
                    break;    
                }
            }
        }
    }
    
    // Reset direction in case of restart
    if ( this->restart_nCG )
    {
        for (int img=0; img<this->noi; img++)
            Engine::Vectormath::set_c_a( 1, { 0, 0, 0 }, this->direction[img] );
    }
};

template <> inline
std::string Method_Solver<Solver::NCG>::SolverName()
{
    return "NCG";
};

template <> inline
std::string Method_Solver<Solver::NCG>::SolverFullName()
{
    return "Nonlinear conjugate gradients";
};