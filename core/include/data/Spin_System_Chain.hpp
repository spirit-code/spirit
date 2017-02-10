#pragma once
#ifndef DATA_SPIN_SYSTEM_CHAIN_H
#define DATA_SPIN_SYSTEM_CHAIN_H

#include "Spirit_Defines.h"
#include <data/Spin_System.hpp>
#include <data/Parameters_Method_GNEB.hpp>

namespace Data
{
	enum class GNEB_Image_Type
	{
		Normal,
		Climbing,
		Falling,
		Stationary
	};

	class Spin_System_Chain
	{
	public:
		// Constructor
		Spin_System_Chain(std::vector<std::shared_ptr<Spin_System>> images, std::shared_ptr<Data::Parameters_Method_GNEB> gneb_parameters, bool iteration_allowed = false);

		int noi;	// Number of Images
		std::vector<std::shared_ptr<Spin_System>> images;
		int idx_active_image;

		// Parameters for GNEB Iterations
		std::shared_ptr<Data::Parameters_Method_GNEB> gneb_parameters;

		// Are we allowed to iterate on this chain?
		bool iteration_allowed;

		// Climbing and falling images
		std::vector<GNEB_Image_Type> image_type;

		// Reaction coordinates of images in the chain
		std::vector<scalar> Rx;

		// Reaction coordinates of interpolated points
		std::vector<scalar> Rx_interpolated;

		// Total Energy of the spin systems and interpolated values
		std::vector<scalar> E_interpolated;
		std::vector<std::vector<scalar>> E_array_interpolated;


	private:
	
	};
}
#endif