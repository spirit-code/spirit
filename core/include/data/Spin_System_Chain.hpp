#pragma once
#ifndef DATA_SPIN_SYSTEM_CHAIN_H
#define DATA_SPIN_SYSTEM_CHAIN_H

#include "Spin_System.hpp"
#include "Parameters_Method_GNEB.hpp"

namespace Data
{
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
		std::vector<bool> climbing_image;
		std::vector<bool> falling_image;

		// Reaction coordinates of images in the chain
		std::vector<double> Rx;

		// Reaction coordinates of interpolated points
		std::vector<double> Rx_interpolated;

		// Total Energy of the spin systems and interpolated values
		std::vector<double> E_interpolated;
		std::vector<std::vector<double>> E_array_interpolated;


	private:
	
	};
}
#endif