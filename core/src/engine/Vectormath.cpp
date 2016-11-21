#include "engine/Vectormath.hpp"

std::vector<scalar> Engine::scalar_product(std::vector<Vector3> vector_v1, std::vector<Vector3> vector_v2)
{
    std::vector<scalar> result(vector_v1.size());
    for (unsigned int i=0; i<vector_v1.size(); ++i)
    {
        result[i] = vector_v1[i].dot(vector_v2[i]);
    }
	return result;
}