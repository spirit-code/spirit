#include "Cubic_Hermite_Spline.h"
#include <cmath>

namespace Utility
{
	namespace Cubic_Hermite_Spline
	{
		// see http://de.wikipedia.org/wiki/Kubisch_Hermitescher_Spline
		std::vector<std::vector<double>> Interpolate(std::vector<double> x, std::vector<double> p, std::vector<double> m, int n_interpolations)
		{
			double x0, x1, p0, p1, m0, m1, t, h00, h10, h01, h11;
			int idx;

			std::vector<std::vector<double>> result(2, std::vector<double>((p.size()-1)*n_interpolations));

			for (unsigned int i = 0; i < p.size()-1; ++i)
			{
				x0 = x[i];
				x1 = x[i + 1];
				p0 = p[i];
				p1 = p[i + 1];
				m0 = m[i];
				m1 = m[i + 1];
				
				for (int j = 0; j < n_interpolations; ++j)
				{
					t = j / (double)n_interpolations;
					h00 = 2*std::pow(t,3) - 3 * std::pow(t,2) + 1;
					h10 = -2 * std::pow(t, 3) + 3 * std::pow(t, 2);
					h01 = std::pow(t, 3) - 2 * std::pow(t, 2) + t;
					h11 = std::pow(t, 3) - std::pow(t, 2);

					idx = i * n_interpolations + j;
					result[0][idx] = x0 + t*(x1-x0);
					result[1][idx] = h00*p0 + h10*p1 + h01*m0 * (x0 - x1) + h11*m1 * (x0 - x1);
				}
			}
			result[0][result[0].size() - 1] = x[x.size() - 1];
			result[1][result[1].size() - 1] = p[p.size() - 1];

			return result;
		}
	}//end namespace Cubic_Hermite_Spline
}//end namespace Utility