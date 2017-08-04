template <> inline
void Method_Template<Solver::SIB>::Solver_Initialise ()
{
    this->forces = std::vector<vectorfield>(this->noi, vectorfield(this->nos));  // [noi][nos]
    this->forces_virtual = std::vector<vectorfield>(this->noi, vectorfield(this->nos));  // [noi][nos]
    
    this->configurations_temp = std::vector<std::shared_ptr<vectorfield>>(this->noi);
    for (int i=0; i<this->noi; ++i) configurations_temp[i] = std::shared_ptr<vectorfield>(new vectorfield(this->nos)); // [noi][nos]
};


/*
    Template instantiation of the Simulation class for use with the SIB Solver.
    The semi-implicit method B is an efficient midpoint solver.
    TODO: reference paper
*/
template <> inline
void Method_Template<Solver::SIB>::Solver_Iteration ()
{
    // First part of the step
    this->Calculate_Force(this->configurations, forces);
    for (int i = 0; i < this->noi; ++i)
    {
        auto& system     = this->systems[i];
        auto& image      = *system->spins;
        auto& image_temp = *configurations_temp[i];
        auto& parameters = *system->llg_parameters;

        this->VirtualForce(image, parameters, forces[i], forces_virtual[i]);
        Vectormath::transform(image, forces_virtual[i], image_temp);
        Vectormath::add_c_a(1, image, image_temp);
        Vectormath::scale(image_temp, 0.5);
    }

    // Second part of the step
    this->Calculate_Force(this->configurations_temp, forces);
    for (int i = 0; i < this->noi; ++i)
    {
        auto& system     = this->systems[i];
        auto& image      = *system->spins;
        auto& image_temp = *configurations_temp[i];
        auto& parameters = *system->llg_parameters;
        
        this->VirtualForce(image_temp, parameters, forces[i], forces_virtual[i]);
        Vectormath::transform(image, forces_virtual[i], image);
    }
};

template <> inline
std::string Method_Template<Solver::SIB>::SolverName()
{
	return "SIB";
};

template <> inline
std::string Method_Template<Solver::SIB>::SolverFullName()
{
	return "Semi-implicit B";
};