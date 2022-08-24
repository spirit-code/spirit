#include <fmt/format.h>

template<>
inline void Method_Solver<Solver::ST>::Initialize()
{

}

/*
    Template instantiation of the Simulation class for use with the ST Solver.
        The semi-implicit method B is an efficient midpoint solver.
    Paper: J. H. Mentink et al., Stable and fast semi-implicit integration of the stochastic
           Landau-Lifshitz equation, J. Phys. Condens. Matter 22, 176001 (2010).
*/
template<>
inline void Method_Solver<Solver::ST>::Iteration()
{
   fmt::print("ST");
}

template<>
inline std::string Method_Solver<Solver::ST>::SolverName()
{
    return "ST";
}

template<>
inline std::string Method_Solver<Solver::ST>::SolverFullName()
{
    return "Suzuki_Trotter";
}