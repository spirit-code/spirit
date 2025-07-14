#pragma once

#include <Spirit/Simulation.h>
#include <utility/Enum.hpp>

#include <array>

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

inline constexpr auto enum_table( Solver )
{
    using E = Utility::Enum::TableElementType<Solver>;
    return std::array{
        // clang-format off
        E{ Solver::None,        "none",        "None",        "None" },
        E{ Solver::SIB,         "sib",         "SIB",         "Semi-implicit B" },
        E{ Solver::Heun,        "heun",        "Heun",        "Heun" },
        E{ Solver::Depondt,     "depondt",     "Depondt",     "Depondt" },
        E{ Solver::RungeKutta4, "rk4",         "RK4",         "Runge Kutta (4th order)" },
        E{ Solver::LBFGS_OSO,   "lbfgs_oso",   "LBFGS_OSO",   "Limited memory Broyden-Fletcher-Goldfarb-Shanno using exponential transforms" },
        E{ Solver::LBFGS_Atlas, "lbfgs_atlas", "LBFGS_Atlas", "Limited memory Broyden-Fletcher-Goldfarb-Shanno using stereographic atlas" },
        E{ Solver::VP,          "vp",          "VP",          "Velocity Projection" },
        E{ Solver::VP_OSO,      "vp_oso",      "VP_OSO",      "Velocity Projection using exponential transforms" },
        // clang-format on
    };
}

} // namespace Common

} // namespace Engine
