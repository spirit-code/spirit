template <>
void Method_Template<Solver::SIB>::Solver_Init ()
{
    std::cerr << "SIB INIT" << std::endl;

    this->xi = vectorfield(this->nos, {0,0,0});
    this->virtualforce = std::vector<vectorfield>(this->noi, vectorfield(this->nos));  // [noi][nos]
    
    this->spins_temp = std::vector<std::shared_ptr<vectorfield>>(this->noi);
    for (int i=0; i<this->noi; ++i) spins_temp[i] = std::shared_ptr<vectorfield>(new vectorfield(this->nos)); // [noi][nos]
};


/*
    Template instantiation of the Simulation class for use with the SIB Solver
*/
template <>
void Method_Template<Solver::SIB>::Solver_Step ()
{
    std::cerr << "SIB STEP" << std::endl;

    std::shared_ptr<Data::Spin_System> s;

    // Random Numbers
    for (int i = 0; i < this->noi; ++i)
    {
        s = this->systems[i];
        if (s->llg_parameters->temperature > 0)
        {
            this->epsilon = std::sqrt(2.0*s->llg_parameters->damping / (1.0 + std::pow(s->llg_parameters->damping, 2))*s->llg_parameters->temperature*Utility::Constants::k_B);
            // Precalculate RNs --> move this up into Iterate and add array dimension n for no of iterations?
            Vectormath::get_random_vectorfield(*s, epsilon, xi);
        }
    }

    // First part of the step
    this->Calculate_Force(configurations, force);
    for (int i = 0; i < this->noi; ++i)
    {
        s = this->systems[i];
        this->VirtualForce(*s->spins, *s->llg_parameters, force[i], xi, virtualforce[i]);
        Vectormath::transform(*s->spins, virtualforce[i], *spins_temp[i]);
        Vectormath::add_c_a(1, *s->spins, *spins_temp[i]);
        Vectormath::scale(*spins_temp[i], 0.5);
    }

    // Second part of the step
    this->Calculate_Force(this->spins_temp, force); ////// I cannot see a difference if this step is included or not...
    for (int i = 0; i < this->noi; ++i)
    {
        s = this->systems[i];
        this->VirtualForce(*spins_temp[i], *s->llg_parameters, force[i], xi, virtualforce[i]);
        Vectormath::transform(*s->spins, virtualforce[i], *s->spins);
    }
};