template <> inline
void Method_Solver<Solver::SIB>::Initialize ()
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
void Method_Solver<Solver::SIB>::Iteration ()
{
    // First part of the step
    this->Calculate_Force_Virtual(this->configurations, forces_virtual);
    for (int i = 0; i < this->noi; ++i)
    {
        auto& image      = *this->systems[i]->spins;
        auto& image_temp = *configurations_temp[i];

        Vectormath::transform(image, forces_virtual[i], image_temp);
        Vectormath::add_c_a(1, image, image_temp);
        Vectormath::scale(image_temp, 0.5);
    }

    // Second part of the step
    this->Calculate_Force_Virtual(this->configurations_temp, forces_virtual);
    for (int i = 0; i < this->noi; ++i)
    {
        auto& image      = *this->systems[i]->spins;
        
        Vectormath::transform(image, forces_virtual[i], image);
    }
};

template <> inline
std::string Method_Solver<Solver::SIB>::SolverName()
{
	return "SIB";
};

template <> inline
std::string Method_Solver<Solver::SIB>::SolverFullName()
{
	return "Semi-implicit B";
};