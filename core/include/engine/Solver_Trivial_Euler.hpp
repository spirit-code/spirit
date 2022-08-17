#include <engine/Backend_par.hpp>

template<>
inline void Method_Solver<Solver::Trivial_Euler>::Initialize()
{
    this->forces         = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->forces_virtual = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->searchdir = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) ); // [noi]
}

template<>
inline void Method_Solver<Solver::Trivial_Euler>::Iteration()
{
    // Get the forces on the configurations
    this->Calculate_Force( configurations, forces );
    this->Calculate_Force_Virtual( configurations, forces, forces_virtual );

    for( int i = 0; i < noi; ++i )
    {
        Backend::par::apply(
            forces[i].size(),
            [sd = this->searchdir[i].data(), dt = this->systems[i]->llg_parameters->dt, f=this->forces[i].data(), m = this->m] SPIRIT_LAMBDA( int idx )
            {
                sd[idx] = 1.0 * dt * f[idx];
            } 
        );
    }
    linesearch->run(configurations, searchdir);
}

template<>
inline std::string Method_Solver<Solver::Trivial_Euler>::SolverName()
{
    return "Trivial_Euler";
}

template<>
inline std::string Method_Solver<Solver::Trivial_Euler>::SolverFullName()
{
    return "Trivial Euler";
}