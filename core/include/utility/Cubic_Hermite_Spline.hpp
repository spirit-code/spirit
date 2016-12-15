#pragma once
#ifndef UTILITY_CUBIC_HERMITE_SPLINE_H
#define UTILITY_CUBIC_HERMITE_SPLINE_H

#include "Core_Defines.h"

#include <vector>

namespace Utility
{
	namespace Cubic_Hermite_Spline
	{
		// Interplation by cubic Hermite spline, see http://de.wikipedia.org/wiki/Kubisch_Hermitescher_Spline
		std::vector<std::vector<scalar>> Interpolate(std::vector<scalar> x, std::vector<scalar> p, std::vector<scalar> m, int n_interpolations);
		
	};//end namespace Cubic_Hermite_Spline
}//end namespace Utility

#endif
