#pragma once
#ifndef DATA_PARAMETERS_METHOD_H
#define DATA_PARAMETERS_METHOD_H

#include "Spirit_Defines.h"
#include <data/Pinning.hpp>

#include <string>
#include <array>
#include <random>

namespace Data
{
	// Solver Parameters Base Class
	class Parameters_Method
	{
	public:
		// Constructor
		Parameters_Method(std::string output_folder, std::string output_file_tag,
            std::array<bool,3> output, long int n_iterations, long int n_iterations_log, 
            long int max_walltime_sec, std::shared_ptr<Pinning> pinning, scalar force_convergence );

		// Data output folder
		std::string output_folder;

		// Put a tag in front of output files (if "<time>" is used then the tag is the timestamp)
		std::string output_file_tag;
		// Save any output when logging
		bool output_any;
		// Save output at initial state
		bool output_initial;
		// Save output at final state
		bool output_final;

		// Maximum walltime for Iterate in seconds
		long int max_walltime_sec;

		// Number of iterations carried out when pressing "play" or calling "iterate"
		long int n_iterations;
		// Number of iterations after which the Method should save data
		long int n_iterations_log;

		// Info on pinned spins
		std::shared_ptr<Pinning> pinning;

		// Force convergence criterium
		scalar force_convergence;
	};
}
#endif