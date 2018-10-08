#pragma once
#ifndef UTILITY_CUBIC_HERMITE_SPLINE_H
#define UTILITY_CUBIC_HERMITE_SPLINE_H

#include "Spirit_Defines.h"

#include <vector>

namespace Utility
{
    namespace Cubic_Hermite_Spline
    {
        // Interplation by cubic Hermite spline, see http://de.wikipedia.org/wiki/Kubisch_Hermitescher_Spline
        std::vector<std::vector<scalar>> Interpolate(const std::vector<scalar> & x, const std::vector<scalar> & p, const std::vector<scalar> & m, int n_interpolations);
    };//end namespace Cubic_Hermite_Spline
}//end namespace Utility

#endif
