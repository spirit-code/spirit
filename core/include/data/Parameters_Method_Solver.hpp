#pragma once
#ifndef SPIRIT_CORE_DATA_PARAMETERS_METHOD_SOLVER_HPP
#define SPIRIT_CORE_DATA_PARAMETERS_METHOD_SOLVER_HPP

#include <data/Parameters_Method.hpp>

namespace Spirit::Data
{

// Solver Parameters Base Class
struct Parameters_Method_Solver : Parameters_Method
{
    // Time step per iteration [ps]
    double dt = 1e-3;
};

} // namespace Spirit::Data

#endif