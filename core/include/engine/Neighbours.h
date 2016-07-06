#pragma once
#ifndef NEIGHBOURS_H
#define NEIGHBOURS_H

#include "Geometry.h"
#include <vector>
#include <memory>

namespace Engine
{
	namespace Neighbours
	{
		// creates Neighbours
		void Create_Neighbours(Data::Geometry & geometry, const std::vector<bool> & boundary_conditions,
			const int n_shells, std::vector<std::vector<int>> &n_spins_in_shell,
			std::vector<std::vector<std::vector<int>>> & neigh,
			std::vector<int> &n_4spin, int &max_n_4spin, std::vector<std::vector<std::vector<int>>> &neigh_4spin,
			std::vector<std::vector<std::vector<double>>> &dm_normal,
			std::vector<std::vector<int>> &segments, std::vector<std::vector<std::vector<double>>> &segments_pos);

		// calculates shell radii for every shell
		void Get_Shell_Radius(const std::vector<double> a, const std::vector<double> b, const std::vector<double> c,
			const int n_shells, double * &shell_radius);

		// calculates array with maximal (for periodic bc) no of neighbours in each shell
		void Get_MaxNumber_NInShell(const std::vector<double> a, const std::vector<double> b, const std::vector<double> c,
			const int n_shells, const double * shell_radius, int * &max_number_n_in_shell, const bool borderOnly);

		// calculates the neighbours within all the shells
		void Get_Neighbours_in_Shells(const int nos, const int n_shells, const std::vector<std::vector<double>> &spin_pos,
			const double *shell_radius, const int number_b_vectors,
			const std::vector<std::vector<double>> &boundary_vectors, std::vector<std::vector<int>> &n_spins_in_shell,
			std::vector<std::vector<std::vector<int>>> & neigh, std::vector<std::vector<std::vector<std::vector<double>>>> & neigh_pos, const bool borderOnly);

		// calculates the 4Spin Neighbours
		void Get_Neighbour_4Spin(const int nos, const int n_shells, const std::vector<std::vector<std::vector<int>>> & neigh, const std::vector<std::vector<int>> &n_spins_in_shell,
			const int &max_n_4spin, std::vector<int> &n_4spin, std::vector<std::vector<std::vector<int>>> &neigh_4spin);

		// calculates the Bulk DMI vectors
		void Get_DM_Norm_Vector_Bulk(const int nos, const std::vector<std::vector<double>> &spin_pos, const int number_b_vectors,
			const std::vector<std::vector<double>> &boundary_vectors, const int n_shells, const std::vector<std::vector<int>> &n_spins_in_shell,
			const std::vector<std::vector<std::vector<int>>> & neigh, std::vector<std::vector<std::vector<std::vector<double>>>> & neigh_pos,
			const int max_ndm, std::vector<std::vector<std::vector<double>>> &dm_normal);

		// calculates the surface DMI vectors
		void Get_DM_Norm_Vector_Surface(const int nos, const std::vector<std::vector<double>> &spin_pos, const int number_b_vectors,
			const std::vector<std::vector<double>> &boundary_vectors, const int n_shells, const std::vector<std::vector<int>> &n_spins_in_shell,
			const std::vector<std::vector<std::vector<int>>> & neigh, std::vector<std::vector<std::vector<std::vector<double>>>> & neigh_pos,
			const int max_ndm, std::vector<std::vector<std::vector<double>>> &dm_normal);

		// prints DMI vectors to file
		void DM_Norm_Vector_To_File(const int nos, const int n_shells, const std::vector<std::vector<int>> &n_spins_in_shell, const std::vector<std::vector<std::vector<int>>> & neigh,
			const std::vector<std::vector<std::vector<double>>> &dm_normal);

		// calculates the segments
		void Get_Segments(const std::vector<double> a, const std::vector<double> b, const std::vector<double> c, int n_shells, const int nos,
			const std::vector<std::vector<double>> &spin_pos, const std::vector<std::vector<int>> &n_spins_in_shell, 
			const std::vector<std::vector<std::vector<int>>> & neigh, std::vector<std::vector<std::vector<std::vector<double>>>> & neigh_pos,
			std::vector<std::vector<int>> &segments, std::vector<std::vector<std::vector<double>>> &segments_pos);

		void Create_Dipole_Neighbours(Data::Geometry & geometry, std::vector<bool> boundary_conditions,
			const double dd_radius, std::vector<std::vector<int>> & dd_neigh, std::vector<std::vector<std::vector<double>>>& dd_neigh_pos,
			std::vector<std::vector<std::vector<double>>> & dd_normal, std::vector<std::vector<double>> & dd_distance);

		// Creates the boundary vectors for given boundary_conditions and geometry
		void Create_Boundary_Vectors(Data::Geometry & geometry, const std::vector<bool> & boundary_conditions, std::vector<std::vector<double>> & boundary_vectors);

		// Convert a list of neighbour shells into a list of pairs.
		void DD_Pairs_from_Neighbours(const Data::Geometry & geometry, const std::vector<std::vector<int>> & dd_neighbours, const std::vector<std::vector<std::vector<double>>> & dd_neighbours_positions, const std::vector<std::vector<double>> & dd_distance, const std::vector<std::vector<std::vector<double>>> & dd_normal,
			std::vector<std::vector<std::vector<int>>> & DD_indices, std::vector<std::vector<double>> & DD_magnitude, std::vector<std::vector<std::vector<double>>> & DD_normal);
	};// end namespace Neighbours
}// end namespace Engine
#endif