#pragma once

#include <Spirit/Simulation.h>

namespace Engine
{

namespace Common
{

enum struct Solver
{
    None        = -1,
    SIB         = Solver_SIB,
    Heun        = Solver_Heun,
    Depondt     = Solver_Depondt,
    RungeKutta4 = Solver_RungeKutta4,
    LBFGS_OSO   = Solver_LBFGS_OSO,
    LBFGS_Atlas = Solver_LBFGS_Atlas,
    VP          = Solver_VP,
    VP_OSO      = Solver_VP_OSO
};

}

} // namespace Engine
