#pragma once
#ifndef DATA_PARAMETERS_METHOD_SOLVER_H
#define DATA_PARAMETERS_METHOD_SOLVER_H

#include "Spirit_Defines.h"
#include <data/Parameters_Method.hpp>

namespace Data
{
    // Solver Parameters Base Class
    struct Parameters_Method_Solver : public Parameters_Method
    {
        // Time step per iteration [ps]
        scalar dt = 1e-3;
    };
}
#endif