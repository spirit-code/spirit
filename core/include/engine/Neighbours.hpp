#pragma once
#ifndef NEIGHBOURS_H
#define NEIGHBOURS_H

#include "Spirit_Defines.h"
#include <data/Geometry.hpp>
#include <engine/Vectormath_Defines.hpp>

#include <vector>
#include <memory>

namespace Engine
{
	namespace Neighbours
	{
		
		std::vector<scalar> Get_Shell_Radius(const Data::Geometry & geometry, const int n_shells);
		void Pairs_from_Neighbour_Shells(const Data::Geometry & geometry, int nShells, std::vector<int> & shellIndex, pairfield & pairs);
		void Neighbours_from_Shells(const Data::Geometry & geometry, int nShells, neighbourfield & neighbours);

		Vector3 DMI_Normal_from_Pair(const Data::Geometry & geometry, Pair pair, int chirality=1);
		void DDI_Pairs_from_Neighbours(const Data::Geometry & geometry, scalar radius, pairfield & pairs, scalarfield & ddi_magnitude, vectorfield & ddi_normal);

		void Create_Dipole_Pairs(const Data::Geometry & geometry, scalar dd_radius,
		std::vector<indexPairs> & DD_indices, std::vector<scalarfield> & DD_magnitude, std::vector<vectorfield> & DD_normal);
		// void Create_Dipole_Neighbours();

		// Convert a list of neighbour shells into a list of pairs.
		void Create_DD_Pairs_from_Neighbours(const Data::Geometry & geometry, const std::vector<std::vector<int>> & dd_neighbours,
			const std::vector<std::vector<Vector3>> & dd_neighbours_positions, const std::vector<std::vector<scalar>> & dd_distance,
			const std::vector<vectorfield> & dd_normal, std::vector<std::vector<std::vector<int>>> & DD_indices,
			std::vector<std::vector<scalar>> & DD_magnitude, std::vector<vectorfield> & DD_normal);
	};// end namespace Neighbours
}// end namespace Engine
#endif