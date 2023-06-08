#pragma once
#ifndef SPIRIT_CORE_UTILITY_CUBIC_HERMITE_SPLINE_HPP
#define SPIRIT_CORE_UTILITY_CUBIC_HERMITE_SPLINE_HPP

#include <Spirit/Spirit_Defines.h>

#include <vector>

namespace Spirit::Utility::Cubic_Hermite_Spline
{

// Interplation by cubic Hermite spline, see http://de.wikipedia.org/wiki/Kubisch_Hermitescher_Spline
std::vector<std::vector<scalar>> Interpolate(
    const std::vector<scalar> & x, const std::vector<scalar> & p, const std::vector<scalar> & m, int n_interpolations );

} // namespace Spirit::Utility::Cubic_Hermite_Spline

#endif