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

		pairfield Get_Pairs_in_Radius(const Data::Geometry & geometry, scalar radius);

		void Get_Neighbours_in_Shells(const Data::Geometry & geometry, int nShells, pairfield & neighbours, intfield & shells, bool remove_redundant);

		Vector3 DMI_Normal_from_Pair(const Data::Geometry & geometry, const Pair & pair, int chirality=1);
		void DDI_from_Pair(const Data::Geometry & geometry, const Pair & pair, scalar & magnitude, Vector3 & normal);
	};// end namespace Neighbours
}// end namespace Engine
#endif