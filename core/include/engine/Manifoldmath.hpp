#pragma once
#ifndef MANIFOLDMATH_H
#define MANIFOLDMATH_H

#include <vector>
#include <memory>

#include <Eigen/Core>

#include <engine/Vectormath_Defines.hpp>

namespace Engine
{
	namespace Manifoldmath
	{
        // Get the norm of a vectorfield
        scalar norm(const vectorfield & vf);
        // Normalize a vectorfield
        void normalize(vectorfield & vf);

        // TODO: the following functions should maybe be characterised as vectorspace instead of manifold
        // Project v1 to be parallel to v2
        //    This assumes normalized vectorfields
        void project_parallel(vectorfield & vf1, const vectorfield & vf2);
        // Project v1 to be orthogonal to v2
        //    This assumes normalized vectorfields
        void project_orthogonal(vectorfield & vf1, const vectorfield & vf2);
        // Invert v1's component parallel to v2
        //    This assumes normalized vectorfields
        void invert_parallel(vectorfield & vf1, const vectorfield & vf2);
        // Invert v1's component orthogonal to v2
        //    This assumes normalized vectorfields
        void invert_orthogonal(vectorfield & vf1, const vectorfield & vf2);

		// Greatcircle distance between two vectors
		scalar dist_greatcircle(const Vector3 & v1, const Vector3 & v2);
		// Geodesic distance between two vectorfields
		scalar dist_geodesic(const vectorfield & v1, const vectorfield & v2);

		// Calculate the "tangent" vectorfields pointing between a set of configurations
		void Tangents(std::vector<std::shared_ptr<vectorfield>> configurations, const std::vector<scalar> & energies, std::vector<vectorfield> & tangents);
    }
}

#endif