template<>
inline void Method_Solver<Solver::SIB>::Initialize()
{
    this->forces         = std::vector<vectorfield>( this->noi, vectorfield( this->nos ) ); // [noi][nos]
    this->forces_virtual = std::vector<vectorfield>( this->noi, vectorfield( this->nos ) ); // [noi][nos]

    this->forces_predictor         = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->forces_virtual_predictor = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );

    this->configurations_predictor = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for( int i = 0; i < this->noi; i++ )
        configurations_predictor[i] = std::shared_ptr<vectorfield>( new vectorfield( this->nos ) );
};

/*
    Template instantiation of the Simulation class for use with the SIB Solver.
        The semi-implicit method B is an efficient midpoint solver.
    Paper: J. H. Mentink et al., Stable and fast semi-implicit integration of the stochastic
           Landau-Lifshitz equation, J. Phys. Condens. Matter 22, 176001 (2010).
*/
template<>
inline void Method_Solver<Solver::SIB>::Iteration()
{
    // Generate random vectors for this iteration
    this->Prepare_Thermal_Field();

    // First part of the step
    this->Calculate_Force( this->configurations, this->forces );
    this->Calculate_Force_Virtual( this->configurations, this->forces, this->forces_virtual );
    for( int i = 0; i < this->noi; ++i )
    {
        auto & image     = *this->systems[i]->spins;
        auto & predictor = *this->configurations_predictor[i];

        Solver_Kernels::sib_transform( image, forces_virtual[i], predictor );
        Vectormath::add_c_a( 1, image, predictor );
        Vectormath::scale( predictor, 0.5 );
    }

    // Second part of the step
    this->Calculate_Force( this->configurations_predictor, this->forces_predictor );
    this->Calculate_Force_Virtual(
        this->configurations_predictor, this->forces_predictor, this->forces_virtual_predictor );
    for( int i = 0; i < this->noi; ++i )
    {
        auto & image = *this->systems[i]->spins;

        Solver_Kernels::sib_transform( image, forces_virtual_predictor[i], image );
    }
};

template<>
inline std::string Method_Solver<Solver::SIB>::SolverName()
{
    return "SIB";
};

template<>
inline std::string Method_Solver<Solver::SIB>::SolverFullName()
{
    return "Semi-implicit B";
};