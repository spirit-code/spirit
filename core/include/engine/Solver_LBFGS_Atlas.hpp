#pragma once
#ifndef SOLVER_LBFGS_ATLAS_HPP
#define SOLVER_LBFGS_ATLAS_HPP

#include <utility/Constants.hpp>
// #include <utility/Exception.hpp>
#include <algorithm>

using namespace Utility;

template <> inline
void Method_Solver<Solver::LBFGS_Atlas>::Initialize ()
{

};

template <> inline
void Method_Solver<Solver::LBFGS_Atlas>::Iteration()
{

}

template <> inline
std::string Method_Solver<Solver::LBFGS_Atlas>::SolverName()
{
    return "LBFGS_Atlas";
}

template <> inline
std::string Method_Solver<Solver::LBFGS_Atlas>::SolverFullName()
{
    return "Limited memory Broyden-Fletcher-Goldfarb-Shanno";
}

#endif