#pragma once
#ifndef VECTORMATH_NEW_H
#define VECTORMATH_NEW_H

#include <vector>

#include <Eigen/Core>

#include "Vectormath_Defines.hpp"

namespace Engine
{
    std::vector<scalar> scalar_product(std::vector<Vector3> vector_v1, std::vector<Vector3> vector_v2);
}

#endif