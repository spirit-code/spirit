#pragma once
#ifndef SPIRIT_CORE_ENGINE_HAMILTONIAN_DEFINES_HPP
#define SPIRIT_CORE_ENGINE_HAMILTONIAN_DEFINES_HPP

#include <engine/Vectormath_Defines.hpp>

namespace Data
{

template<typename T>
using labeled = std::pair<std::string, T>;

template<typename T>
using vectorlabeled = std::vector<labeled<T>>;

struct NormalVector
{
    scalar magnitude;
    Vector3 normal;
};

struct ScalarfieldData
{
    intfield indices;
    scalarfield magnitudes;
};

struct VectorfieldData : ScalarfieldData
{
    vectorfield normals;
};

struct ScalarPairfieldData
{
    pairfield pairs;
    scalarfield magnitudes;
};

struct VectorPairfieldData : ScalarPairfieldData
{
    vectorfield normals;
};

struct QuadrupletfieldData
{
    quadrupletfield quadruplets;
    scalarfield magnitudes;
};

struct DDI_Data
{
    intfield n_periodic_images;
    scalar radius;
    bool pb_zero_padding;
};

} // namespace Data
#endif
