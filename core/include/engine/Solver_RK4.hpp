template <> inline
void Method_Solver<Solver::RungeKutta4>::Initialize ()
{
    this->forces         = std::vector<vectorfield>( this->noi, vectorfield( this->nos, {0, 0, 0} ) );
    this->forces_virtual = std::vector<vectorfield>( this->noi, vectorfield( this->nos, {0, 0, 0} ) );

    this->forces_predictor = std::vector<vectorfield>( this->noi, vectorfield( this->nos, {0, 0, 0} ) );
    this->forces_virtual_predictor = std::vector<vectorfield>( this->noi, vectorfield( this->nos, {0, 0, 0} ) );

    this->configurations_temp  = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for (int i=0; i<this->noi; i++)
      this->configurations_temp[i] = std::shared_ptr<vectorfield>(new vectorfield(this->nos));

    this->configurations_predictor = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for (int i=0; i<this->noi; i++)
      this->configurations_predictor[i] = std::shared_ptr<vectorfield>(new vectorfield(this->nos));

    this->configurations_k1 = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for (int i=0; i<this->noi; i++)
      this->configurations_k1[i] = std::shared_ptr<vectorfield>(new vectorfield(this->nos));

    this->configurations_k2 = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for (int i=0; i<this->noi; i++)
      this->configurations_k2[i] = std::shared_ptr<vectorfield>(new vectorfield(this->nos));

    this->configurations_k3 = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for (int i=0; i<this->noi; i++)
      this->configurations_k3[i] = std::shared_ptr<vectorfield>(new vectorfield(this->nos));

    this->configurations_k4 = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for (int i=0; i<this->noi; i++)
      this->configurations_k4[i] = std::shared_ptr<vectorfield>(new vectorfield(this->nos));

    this->temp1 = vectorfield( this->nos, {0, 0, 0} );
};


/*
    Template instantiation of the Simulation class for use with the 4th order Runge Kutta Solver.
*/
template <> inline
void Method_Solver<Solver::RungeKutta4>::Iteration ()
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
        auto force          = this->forces_virtual[i].data();
        auto k1             = this->configurations_k1[i]->data();

        Backend::par::apply( nos, [conf, conf_predictor, force, k1] SPIRIT_LAMBDA (int idx) {
            k1[idx] = - conf[idx].cross(force[idx]);
            conf_predictor[idx] = (conf[idx] + 0.5*k1[idx]).normalized();
        } );
    }

    // Calculate_Force for the predictor
    this->Calculate_Force(this->configurations_predictor, this->forces_predictor);
    this->Calculate_Force_Virtual(this->configurations_predictor, this->forces_predictor, this->forces_virtual_predictor);

    // Predictor for each image
    for (int i = 0; i < this->noi; ++i)
    {
        auto conf           = this->configurations[i]->data();
        auto conf_predictor = this->configurations_predictor[i]->data();
        auto force          = this->forces_virtual_predictor[i].data();
        auto k2             = this->configurations_k2[i]->data();

        Backend::par::apply( nos, [conf, conf_predictor, force, k2] SPIRIT_LAMBDA (int idx) {
            k2[idx] = - conf_predictor[idx].cross(force[idx]);
            conf_predictor[idx] = (conf[idx] + 0.5*k2[idx]).normalized();
        } );
    }

    // Calculate_Force for the predictor (k3)
    this->Calculate_Force(this->configurations_predictor, this->forces_predictor);
    this->Calculate_Force_Virtual(this->configurations_predictor, this->forces_predictor, this->forces_virtual_predictor);

    // Predictor for each image
    for (int i = 0; i < this->noi; ++i)
    {
        auto conf           = this->configurations[i]->data();
        auto conf_predictor = this->configurations_predictor[i]->data();
        auto force          = this->forces_virtual_predictor[i].data();
        auto k3             = this->configurations_k3[i]->data();

        Backend::par::apply( nos, [conf, conf_predictor, force, k3] SPIRIT_LAMBDA (int idx) {
            k3[idx] = - conf_predictor[idx].cross(force[idx]);
            conf_predictor[idx] = (conf[idx] + k3[idx]).normalized();
        } );
    }

    // Calculate_Force for the predictor (k4)
    this->Calculate_Force(this->configurations_predictor, this->forces_predictor);
    this->Calculate_Force_Virtual(this->configurations_predictor, this->forces_predictor, this->forces_virtual_predictor);

    // Corrector step for each image
    for (int i=0; i < this->noi; i++)
    {
        auto conf           = this->configurations[i]->data();
        auto conf_predictor = this->configurations_predictor[i]->data();
        auto force          = this->forces_virtual_predictor[i].data();
        auto k1             = this->configurations_k1[i]->data();
        auto k2             = this->configurations_k2[i]->data();
        auto k3             = this->configurations_k3[i]->data();
        auto k4             = this->configurations_k4[i]->data();

        Backend::par::apply( nos, [conf, conf_predictor, force, k1, k2, k3, k4] SPIRIT_LAMBDA (int idx) {
            k4[idx] = - conf_predictor[idx].cross(force[idx]);
            conf[idx] = (conf[idx] + 1.0/6.0*k1[idx] + 1.0/3.0*k2[idx] + 1.0/3.0*k3[idx] + 1.0/6.0*k4[idx]).normalized();
        } );
    }
};

template <> inline
std::string Method_Solver<Solver::RungeKutta4>::SolverName()
{
    return "RK4";
};

template <> inline
std::string Method_Solver<Solver::RungeKutta4>::SolverFullName()
{
    return "Runge Kutta (4th order)";
};