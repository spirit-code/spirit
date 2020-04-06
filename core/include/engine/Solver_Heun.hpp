template <> inline
void Method_Solver<Solver::Heun>::Initialize ()
{
    this->forces         = std::vector<vectorfield>( this->noi, vectorfield( this->nos, {0, 0, 0} ) );
    this->forces_virtual = std::vector<vectorfield>( this->noi, vectorfield( this->nos, {0, 0, 0} ) );

    this->forces_predictor = std::vector<vectorfield>( this->noi, vectorfield( this->nos, {0, 0, 0} ) );
    this->forces_virtual_predictor = std::vector<vectorfield>( this->noi, vectorfield( this->nos, {0, 0, 0} ) );

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
    This method is described for spin systems including thermal noise in
        U. Nowak, Thermally Activated Reversal in Magnetic Nanostructures,
        Annual Reviews of Computational Physics IX Chapter III (p 105) (2001)
*/
template <> inline
void Method_Solver<Solver::Heun>::Iteration ()
{
    // Generate random vectors for this iteration
    this->Prepare_Thermal_Field();

    // Get the actual forces on the configurations
    this->Calculate_Force(this->configurations, this->forces);
    this->Calculate_Force_Virtual(this->configurations, this->forces, this->forces_virtual);

    // Predictor for each image
    for (int i = 0; i < this->noi; ++i)
    {
        auto conf           = this->configurations[i]->data();
        auto conf_predictor = this->configurations_predictor[i]->data();
        auto f_virtual      = this->forces_virtual[i].data();

        // First step - Predictor
        Backend::par::apply( nos, [conf, conf_predictor, f_virtual] SPIRIT_LAMBDA (int idx) {
            conf_predictor[idx] = ( conf[idx] - conf[idx].cross( f_virtual[idx] ) ).normalized();
        } );
    }

    // Calculate_Force for the Corrector
    this->Calculate_Force(this->configurations_predictor, this->forces_predictor);
    this->Calculate_Force_Virtual(this->configurations_predictor, this->forces_predictor, this->forces_virtual_predictor);

    // Corrector step for each image
    for (int i=0; i < this->noi; i++)
    {
        auto conf           = this->configurations[i]->data();
        auto conf_predictor = this->configurations_predictor[i]->data();
        auto f_virtual      = this->forces_virtual[i].data();
        auto f_virtual_predictor = this->forces_virtual_predictor[i].data();

        // Second step - Corrector
        Backend::par::apply( nos, [conf, conf_predictor, f_virtual, f_virtual_predictor] SPIRIT_LAMBDA (int idx) {
            conf[idx] = (
                conf[idx] + 0.5*( conf[idx] - 0.5*conf[idx].cross(f_virtual[idx]) - conf_predictor[idx].cross(f_virtual_predictor[idx]) )
            ).normalized();
        } );
    }
};

template <> inline
std::string Method_Solver<Solver::Heun>::SolverName()
{
    return "Heun";
};

template <> inline
std::string Method_Solver<Solver::Heun>::SolverFullName()
{
    return "Heun";
};