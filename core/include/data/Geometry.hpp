#pragma once
#ifndef DATA_GEOMETRY_H
#define DATA_GEOMETRY_H

#include <vector>

namespace Data
{
	// Geometry contains all geometric information about the spin_system
	// The members are const, as a spin system has to be created new whenever one of these members is changed.
	class Geometry
	{
	public:
		// Constructor
		Geometry(const std::vector<std::vector<double>> basis, const std::vector<std::vector<double>> translation_vectors, 
			const std::vector<int> n_cells, const int n_spins_basic_domain, const std::vector<double> spin_pos);
		// Destructor
		//~Geometry();


		// -------------------- Input constants ------------------
		// Basis [dim][basis_vec]
		const std::vector<std::vector<double>> basis;
		// Translation Vectors [dim][transl_vec]
		const std::vector<std::vector<double>> translation_vectors;
		// Number of Translations {nta, ntb, ntc}
		const std::vector<int> n_cells;
		// Number of spins per basic domain
		const int n_spins_basic_domain;
		// TODO: array of basis atom positions [n][3]
		// std::vector<std::vector<double>> basis_atoms;
		// Number of Spins total
		const int nos;
		// number of shells -> moved to Hamiltonian
		// const int n_shells;
		
		// Center and Bounds
		std::vector<double> center;
		std::vector<double> bounds_min;
		std::vector<double> bounds_max;

		// -------------------- Calculated Geometry ------------------
		// number of boundary vectors
		//const int number_boundary_vectors;
		// Boundary vectors: boundary_vectors[dim][number_b_vectors]
		//const std::vector<std::vector<double>> boundary_vectors;

		// Positions of the Spins: spin_pos[dim][nos]
		const std::vector<double> spin_pos;
						
		// //////  What are the segments used for??
		// segments[nos][4]
		//const std::vector<std::vector<int>> segments;
		// Position of the Segments: segments_pos[dim][nos][4]
		//const std::vector<std::vector<std::vector<double>>> segments_pos;
	};
}
#endif