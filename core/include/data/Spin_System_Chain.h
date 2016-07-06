#pragma once
#ifndef DATA_SPIN_SYSTEM_CHAIN_H
#define DATA_SPIN_SYSTEM_CHAIN_H

#include "Spin_System.h"
#include "Parameters_GNEB.h"

namespace Data
{
	class Spin_System_Chain
	{
	public:
		// Constructor
		Spin_System_Chain(std::vector<std::shared_ptr<Spin_System>> images, std::shared_ptr<Data::Parameters_GNEB> gneb_parameters, bool iteration_allowed = false);

		int noi;	// Number of Images
		std::vector<std::shared_ptr<Spin_System>> images;
		int active_image;

		// Parameters for GNEB Iterations
		std::shared_ptr<Data::Parameters_GNEB> gneb_parameters;

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

		// tangents to be accessed by e.g. solver
		std::vector<std::vector<double>> tangents;

		// Update E and Rx
		void Update_Data();

		// Add an image to the chain, before idx
		void Insert_Image_Before(int idx, std::shared_ptr<Data::Spin_System> system);
		// Add an image to the chain, after idx
		void Insert_Image_After(int idx, std::shared_ptr<Data::Spin_System> system);
		// Replace image at idx
		void Replace_Image(int idx, std::shared_ptr<Data::Spin_System> system);
		// Delete an image
		void Delete_Image(int idx);

	private:
		void Setup_Initial_Data();
	};
}
#endif