#include "engine/Vectormath.hpp"

std::vector<scalar> Engine::Vectormath::scalar_product(std::vector<Vector3> vector_v1, std::vector<Vector3> vector_v2)
{
    std::vector<scalar> result(vector_v1.size());
    for (unsigned int i=0; i<vector_v1.size(); ++i)
    {
        result[i] = vector_v1[i].dot(vector_v2[i]);
    }
	return result;
}


void Engine::Vectormath::Normalize_3Nos(std::vector<Vector3> & spins)
{
	scalar norm = 0;
	for (int i = 0; i < spins.size(); ++i)
	{
		for (int dim = 0; dim < 3; ++dim)
		{
			norm += std::pow(spins[i][dim], 2);
		}
	}
	scalar norm1 = 1.0/norm;
	for (int i = 0; i < spins.size(); ++i)
	{
		spins[i] *= norm1;
	}
}



scalar Engine::Vectormath::dist_greatcircle(Vector3 v1, Vector3 v2)
{
	scalar r = v1.dot(v2);

	// Prevent NaNs from occurring
	r = std::fmax(-1.0, std::fmin(1.0, r));

	// Greatcircle distance
	return std::acos(r);
}