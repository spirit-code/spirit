#pragma once
#ifndef DATA_PINNING_H
#define DATA_PINNING_H

#include <data/Geometry.hpp>
#include <engine/Vectormath_Defines.hpp>

#include <memory>

namespace Data
{
	// Solver Parameters Base Class
	class Pinning
	{
	public:
		// Constructor
		Pinning(std::shared_ptr<Geometry> geometry,
            int na_left, int na_right,
            int nb_left, int nb_right,
            int nc_left, int nc_right,
			vectorfield pinned_cell);

		Pinning(std::shared_ptr<Geometry> geometry,
			intfield mask_unpinned,
			vectorfield mask_pinned_cells);
		
		// Set pinned vectors in a vectorfield
		void Apply(vectorfield & vf);

		//intfield mask_pinned;
		intfield mask_unpinned;
		vectorfield mask_pinned_cells;
	
	private:
		std::shared_ptr<Geometry> geometry;
	};
}
#endif