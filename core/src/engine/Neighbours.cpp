#include "Neighbours.h"
#include "Vectormath.h"
#include "IO.h"
#include "Logging.h"

#include <numeric>
#include <iostream>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include "Exception.h"

using namespace Utility;

namespace Engine
{
	void Neighbours::Create_Neighbours(const Data::Geometry & geometry, const std::vector<bool> & boundary_conditions, 
		const int n_shells, std::vector<std::vector<int>> &n_spins_in_shell,
		std::vector<std::vector<std::vector<int>>> & neigh,
		std::vector<int> &n_4spin, int &max_n_4spin, std::vector<std::vector<std::vector<int>>> &neigh_4spin,
		std::vector<std::vector<std::vector<double>>> &dm_normal,
		std::vector<std::vector<int>> &segments, std::vector<std::vector<std::vector<double>>> &segments_pos)
	{
		//========================= Init local vars ================================
		auto nos = geometry.nos;
		//int max_neigh_number = 0, max_nt4_spin, max_ndm, ispin, jneigh;
		int max_ndm, i, j;
		double spin_sum_3d = 0;
		std::vector<int> max_number_n_in_shell;
		std::vector<double> shell_radius;
		std::vector<std::vector<double>> boundary_vectors;
		int n_boundary_vectors;
		std::vector<double> build_array = { 0.0, 0.0, 0.0 };
		//------------------------ End Init ----------------------------------------
		
		// Calculate boundary vectors
		boundary_vectors = Get_Boundary_Vectors(geometry, boundary_conditions);
		n_boundary_vectors = boundary_vectors[0].size();

		// Calculate shell radii
		shell_radius = Get_Shell_Radius(geometry, n_shells);
		// Check if domain makes sense
		for (i = 1; i < n_boundary_vectors; ++i) {
			for (j = 0; j < n_shells; ++j) {
				if (2 * shell_radius[j] >= Vectormath::Length(boundary_vectors, i))
				{
					Log.Send(Log_Level::SEVERE, Log_Sender::ALL, "Simulated domain is too small!\n     Shell " + std::to_string(n_shells) + " diameter=" + std::to_string(2 * shell_radius[j])
					+ "\n    B.V.  " + std::to_string(i) + "   length=" + std::to_string(Vectormath::Length(boundary_vectors, i))
					+ "\n    B.V.  " + std::to_string(i) + "   vec   =" + std::to_string(boundary_vectors[0][i]) + " " + std::to_string(boundary_vectors[1][i]) + " " + std::to_string(boundary_vectors[2][i]));
					throw Exception::Simulated_domain_too_small;
				}//endif shell_radius >= Length(boundary_vecs)
			}//endfor jneigh
		}//endfor ispin

		// Maximum number of neighbours in any shell
		max_number_n_in_shell = Get_MaxNumber_NInShell(geometry, n_shells, shell_radius, boundary_vectors, true);
		int max_number_n = 0;
		for (i = 0; i < n_shells; ++i) {
			if (max_number_n < max_number_n_in_shell[i]) {
				max_number_n = max_number_n_in_shell[i];
			}
		}


		// Allocating Vectors
		n_spins_in_shell = std::vector<std::vector<int>>(geometry.nos, std::vector<int>(n_shells));
		segments = std::vector<std::vector<int>>(geometry.nos, std::vector<int>(4));
		segments_pos = std::vector<std::vector<std::vector<double>>>(3, std::vector<std::vector<double>>(geometry.nos, std::vector<double>(4)));
		n_4spin = std::vector<int>(geometry.nos);

		// Calculate the Neighbours and their Positions in the shells
		neigh = std::vector<std::vector<std::vector<int>>>(nos, std::vector<std::vector<int>>(n_shells, std::vector<int>(max_number_n))); // neigh[nos][n_shells][max_number_n_in_shell[n_shells]]
		auto neigh_pos = std::vector<std::vector<std::vector<std::vector<double>>>>(3, std::vector<std::vector<std::vector<double>>>(nos, std::vector<std::vector<double>>(n_shells, std::vector<double>(max_number_n))));
		Get_Neighbours_in_Shells(geometry, n_shells, shell_radius, boundary_vectors, n_spins_in_shell, neigh, neigh_pos, true);

		// What do we need this for??
		Create_Segments(geometry, n_shells, nos, geometry.spin_pos, n_spins_in_shell, neigh, neigh_pos, segments, segments_pos);

		// Four-Spin Neighbours
		for (i = 0; i < nos; ++i) {
			spin_sum_3d += std::abs(geometry.spin_pos[2][i]);
		}
		if (spin_sum_3d == 0)
		{
			Log.Send(Log_Level::INFO, Log_Sender::ALL, "------------- 2D Case -------------");
			switch (max_number_n_in_shell[0]) {

			case 4: // square lattice
				max_n_4spin = 4; //diamonds
				Log.Send(Log_Level::INFO, Log_Sender::ALL, "4-spin for square lattice!");
				neigh_4spin = std::vector<std::vector<std::vector<int>>>(3, std::vector<std::vector<int>>(nos, std::vector<int>(max_n_4spin)));
				Create_Neighbours_4Spin(nos, n_shells, neigh, n_spins_in_shell, max_n_4spin, n_4spin, neigh_4spin);
				break;

			case 6: // hexagonal lattice
				max_n_4spin = 12; //diamonds
				Log.Send(Log_Level::INFO, Log_Sender::ALL, "4-spin for hexagonal lattice!");
				neigh_4spin = std::vector<std::vector<std::vector<int>>>(3, std::vector<std::vector<int>>(nos, std::vector<int>(max_n_4spin)));
				Create_Neighbours_4Spin(nos, n_shells, neigh, n_spins_in_shell, max_n_4spin, n_4spin, neigh_4spin);
				break;

			default:
				max_n_4spin = 1;
				Log.Send(Log_Level::WARNING, Log_Sender::ALL, "4-spin interaction is not defined for this type of lattice!");
				// Maybe need to somehow tell the System that K_ijkl should be zero
				//kijkl = 0;
			}//end switch
		}//end 2D-Case
		else
		{
			Log.Send(Log_Level::INFO, Log_Sender::ALL, "------------- 3D Case -------------");
			Log.Send(Log_Level::WARNING, Log_Sender::ALL, "4-spin interaction is not defined for the 3D case");
			// Maybe need to somehow tell the System that K_ijkl should be zero
			//kijkl = 0;
		}//end 3D-Case

		// Maximum number of DM interactions per spin (max_ndm).
		//		Coincides with number of neighbours in the 1-st shell (nearest neighbours)
		max_ndm = max_number_n_in_shell[0];
		// Calculate DM normal vectors
		dm_normal = std::vector<std::vector<std::vector<double>>>(3, std::vector<std::vector<double>>(nos, std::vector<double>(max_ndm)));
		Create_DM_Norm_Vectors_Bulk(nos, geometry.spin_pos, n_boundary_vectors, boundary_vectors, n_shells, n_spins_in_shell, neigh, neigh_pos, max_ndm, dm_normal);
		//DM_Norm_Vectors_To_File(nos, n_shells, n_spins_in_shell, neigh, dm_normal);

		Log.Send(Log_Level::INFO, Log_Sender::ALL, "Done creating Neighbours");
	}//end Neighbours::Create_Neighbours


	std::vector<double> Neighbours::Get_Shell_Radius(const Data::Geometry & geometry, const int n_shells)
	{
		auto shell_radius = std::vector<double>(n_shells);
		//========================= Init local vars ================================
		std::vector<double> a = { geometry.basis[0][0], geometry.basis[1][0], geometry.basis[2][0] };
		std::vector<double> b = { geometry.basis[0][1], geometry.basis[1][1], geometry.basis[2][1] };
		std::vector<double> c = { geometry.basis[0][2], geometry.basis[1][2], geometry.basis[2][2] };

		double radius_current_shell, radius_test_shell, dist, min_shell_distance;
		int i = 0, j = 0, k = 0;
		int ii, jj, kk;
		//the 10 is a value that is big enough by experience to 
		// produce enough needed shells, but is small enough to run sufficiently quick
		int imax = 10, jmax = 10, kmax = 10;
		std::vector<double> build_array = { 0.0, 0.0, 0.0 };

		//abort condidions for all 3 vectors
		if (Vectormath::Length(a) == 0.0) imax = 0;
		if (Vectormath::Length(b) == 0.0) jmax = 0;
		if (Vectormath::Length(c) == 0.0) kmax = 0;

		std::string output_to_file;

		const int buffer_length = 45;
		output_to_file.reserve(n_shells * buffer_length + 2);	//memory allocation for fast append
		char buffer_string_conversion[buffer_length + 2];

		//------------------------ End Init ----------------------------------------

		for (int n = 0; n < n_shells; ++n)
		{

			min_shell_distance = 100.0;
			Vectormath::Array_Array_Add(a, b, c, build_array, i, j, k);
			radius_current_shell = Vectormath::Length(build_array);
			for (ii = imax; ii >= -imax; --ii) {
				for (jj = jmax; jj >= -jmax; --jj) {
					for (kk = kmax; kk >= -kmax; --kk) {
						Vectormath::Array_Array_Add(a, b, c, build_array, ii, jj, kk);
						radius_test_shell = Vectormath::Length(build_array);
						dist = radius_test_shell - radius_current_shell;
						if ((dist > 1.0E-6) && (dist <= min_shell_distance)) {
							min_shell_distance = dist;
							i = ii; j = jj; k = kk;
						}//endif
					}//endfor kk
				}//endfor jj
			}//endfor ii
			Vectormath::Array_Array_Add(a, b, c, build_array, i, j, k);

			shell_radius[n] = Vectormath::Length(build_array);

			//convert values to string and concernate string for filedump
			snprintf(buffer_string_conversion, buffer_length, "%+05i  %+13.6f  %+05i  %+05i  %+05i \n", n, shell_radius[n], i, j, k);
			output_to_file.append(buffer_string_conversion);
		}//endfor n
		IO::Dump_to_File(output_to_file, "output/Shell_Radius.dat");
		return shell_radius;
	}//end Neighbours::Get_Shell_Radius


	std::vector<int> Neighbours::Get_MaxNumber_NInShell(const Data::Geometry & geometry, const int n_shells, const std::vector<double> & shell_radius, 
		std::vector<std::vector<double>> boundary_vectors, const bool borderOnly)
	{
		auto max_number_n_in_shell = std::vector<int>(n_shells, 0);
		auto max_temp = std::vector<int>(n_shells, 0);
		//========================= Init local vars ================================
		std::vector<double> ipos = { 0.0, 0.0, 0.0 };
		std::vector<double> jpos = { 0.0, 0.0, 0.0 };
		double dist;
		unsigned int bvector, number_b_vectors = boundary_vectors[0].size();
		int iatom, jatom, shell;
		std::vector<double> build_array = { 0.0, 0.0, 0.0 };

		int imax = 10, jmax = 10, kmax = 10;
		if (!borderOnly) {
			// Set the n_cells in which to fit this shell and determine MaxNumber_NInShell
			imax = 30; jmax = 30; kmax = 30;
			Log.Send(Utility::Log_Level::DEBUG, Utility::Log_Sender::ALL, "Remember Maximum size of Dipole shell should fit into " + std::to_string(imax) + "x" + std::to_string(jmax) + "x" + std::to_string(kmax) + " translations. Neighbours.cpp line 178 to change this");
		}
		// Retrieve basis vectors from geometry.basis
		std::vector<double> a = { geometry.basis[0][0], geometry.basis[1][0], geometry.basis[2][0] };
		std::vector<double> b = { geometry.basis[0][1], geometry.basis[1][1], geometry.basis[2][1] };
		std::vector<double> c = { geometry.basis[0][2], geometry.basis[1][2], geometry.basis[2][2] };
		//abort condidions for all 3 vectors
		if (Vectormath::Length(a) == 0.0) imax = 0;
		if (Vectormath::Length(b) == 0.0) jmax = 0;
		if (Vectormath::Length(c) == 0.0) kmax = 0;
		//------------------------ End Init ----------------------------------------
		// Create stub for function to check wether the atom is in the shell
		std::function<bool(double)> check = [](double d) { return false; };
		if (borderOnly) 
		{
			// Consider only the atoms on a specific distance d
			check = [](double d)
			{
				return std::abs(d) < 1.0E-5;
        	};
		}
		else
		{
			// Consider all the atoms inside a distance d
			check = [](double d)
			{
				return d < 1.0E-5;
        	};
		}
		// Loop over all atoms
		for (iatom = 0; iatom < 1; ++iatom)//geometry.nos; ++iatom)
		{
			for (shell = 0; shell < n_shells; ++shell) max_temp[shell] = 0;
			ipos[0] = geometry.spin_pos[0][iatom];
			ipos[1] = geometry.spin_pos[1][iatom];
			ipos[2] = geometry.spin_pos[2][iatom];
			// Loop over neighbours, avoid double counting
			for (jatom = iatom + 1; jatom < geometry.nos; ++jatom)
			{
				// Loop also over atoms translated by boundary conditions
				for (bvector = 0; bvector < number_b_vectors; ++bvector)
				{
					jpos[0] = geometry.spin_pos[0][jatom] + boundary_vectors[0][bvector];
					jpos[1] = geometry.spin_pos[1][jatom] + boundary_vectors[1][bvector];
					jpos[2] = geometry.spin_pos[2][jatom] + boundary_vectors[2][bvector];
					// ipos - jpos
					Vectormath::Array_Array_Add(ipos, jpos, build_array, 1.0, -1.0);
					// dist = |ipos-jpos|
					dist = Vectormath::Length(build_array);
					// Test if the distance corresponds to a shell
					for (shell = 0; shell < n_shells; ++shell)
					{
						if ( check(dist - shell_radius[shell]) )
						{
							++max_temp[shell];
							goto B;
						}
					}
				}
			B:;
			}
			// Test if current atom has a greater number of neighbours than the previous ones
			for (shell = 0; shell < n_shells; ++shell)
			{
				if (max_temp[shell] > max_number_n_in_shell[shell]) max_number_n_in_shell[shell] = max_temp[shell];
			}
		}
		// Return
		return max_number_n_in_shell;
	}//end Neighbours::Get_MaxNumber_NInShell


	void Neighbours::Get_Neighbours_in_Shells(const Data::Geometry & geometry, const int n_shells, const std::vector<double> & shell_radius,
		const std::vector<std::vector<double>> &boundary_vectors, std::vector<std::vector<int>> &n_spins_in_shell,
		std::vector<std::vector<std::vector<int>>> & neigh, std::vector<std::vector<std::vector<std::vector<double>>>> & neigh_pos, const bool borderOnly)
	{
		//========================= Init local vars ================================
		std::vector<double> ipos = { 0.0, 0.0, 0.0 };
		std::vector<double> jpos = { 0.0, 0.0, 0.0 };
		double dist;
		unsigned int bvector, number_b_vectors = boundary_vectors[0].size();
		int iatom, jatom, jspin, shell;
		std::vector<double> build_array = { 0.0, 0.0, 0.0 };

		std::string output_to_file;
		if (geometry.nos < 5E+06) {											// if nos too big - guard preallocation against int overflow
			output_to_file.reserve(geometry.nos * 22 + geometry.nos * 6 * 20 + 200);	//memory allocation for fast append
		}
		else { output_to_file.reserve(int(1E+08)); }

		const int buffer_length = 80, name_length = 29;
		char buffer_file_name[name_length + 2];
		char buffer_string_conversion[buffer_length + 2];
		//------------------------ End Init ----------------------------------------
		// Create stub for function to check wether the atom is in the shell
		std::function<bool(double)> check = [](double d) { return false; };
		if (borderOnly) 
		{
			// Consider only the atoms on a specific distance d
			check = [](double d)
			{
				return std::abs(d) < 1.0E-5;
        	};
		}
		else
		{
			// Consider all the atoms inside a distance d
			check = [](double d)
			{
				return d < 1.0E-5;
        	};
		}
		// Loop over all atoms
		for (iatom = 0; iatom < geometry.nos; ++iatom)
		{
			ipos[0] = geometry.spin_pos[0][iatom];
			ipos[1] = geometry.spin_pos[1][iatom];
			ipos[2] = geometry.spin_pos[2][iatom];
			// Loop over neighbours, avoid double counting
			for (jatom = iatom + 1; jatom < geometry.nos; ++jatom)
			{
				// Loop also over atoms translated by boundary conditions
				for (bvector = 0; bvector < number_b_vectors; ++bvector)
				{
					jpos[0] = geometry.spin_pos[0][jatom] + boundary_vectors[0][bvector];
					jpos[1] = geometry.spin_pos[1][jatom] + boundary_vectors[1][bvector];
					jpos[2] = geometry.spin_pos[2][jatom] + boundary_vectors[2][bvector];

					// ipos - jpos
					Vectormath::Array_Array_Add(ipos, jpos, build_array, 1.0, -1.0);
					// dist = |ipos-jpos|
					dist = Vectormath::Length(build_array);
					// if (dist < 3.0) /* Only take Neighbours with dist < ... to save computation time. This should not be done with long-range interactions! */
					// {
					for (shell = 0; shell < n_shells; ++shell)
					{
						// Test if the distance corresponds to a shell
						if ( check(dist - shell_radius[shell]) )
						{
							neigh[iatom][shell][n_spins_in_shell[iatom][shell]] = jatom;
							// Vectormath::Vector_Copy_i(neigh_pos, jpos, 0, iatom, shell, n_spins_in_shell[iatom][shell]);
							neigh_pos[0][iatom][shell][n_spins_in_shell[iatom][shell]] = jpos[0];
							neigh_pos[1][iatom][shell][n_spins_in_shell[iatom][shell]] = jpos[1];
							neigh_pos[2][iatom][shell][n_spins_in_shell[iatom][shell]] = jpos[2];

							n_spins_in_shell[iatom][shell] = n_spins_in_shell[iatom][shell] + 1;

							neigh[jatom][shell][n_spins_in_shell[jatom][shell]] = iatom;
							neigh_pos[0][jatom][shell][n_spins_in_shell[jatom][shell]] = ipos[0] - boundary_vectors[0][bvector];
							neigh_pos[1][jatom][shell][n_spins_in_shell[jatom][shell]] = ipos[1] - boundary_vectors[1][bvector];
							neigh_pos[2][jatom][shell][n_spins_in_shell[jatom][shell]] = ipos[2] - boundary_vectors[2][bvector];
							n_spins_in_shell[jatom][shell] = n_spins_in_shell[jatom][shell] + 1;
							goto A;
						}//endif abs(dist-shell_radius) < 1.0E-5
					}//endfor shell
					// }//endif dist < 3.0
				}//endfor bvector
			A:;
			}//endfor jatom
		}//endfor iatom

		for (shell = 0; shell < n_shells; ++shell)
		{
			snprintf(buffer_file_name, name_length, "output/neighbours_in_shell_%03i.dat", shell + 1);
			output_to_file = "List of neighbours in the ";
			snprintf(buffer_string_conversion, buffer_length, "%03i-th shell\n Spin   Number of |List of   \n number|neighbours|neighbours", shell + 1);
			output_to_file.append(buffer_string_conversion);
			for (iatom = 0; iatom < geometry.nos; ++iatom)
			{
				snprintf(buffer_string_conversion, buffer_length, "\n%7i| %7i  |", iatom, n_spins_in_shell[iatom][shell]);
				output_to_file.append(buffer_string_conversion);
				for (jatom = 0; jatom < n_spins_in_shell[iatom][shell]; ++jatom)
				{
					snprintf(buffer_string_conversion, buffer_length, "%6i", neigh[iatom][shell][jatom]);
					output_to_file.append(buffer_string_conversion);
				}//endfor jatom
			}//endfor iatom
			IO::Dump_to_File(output_to_file, buffer_file_name);
		}//endfor shell

		output_to_file = "";
		for (iatom = 0; iatom < geometry.nos; ++iatom)
		{
			snprintf(buffer_string_conversion, buffer_length, "\n %06i|", iatom);
			output_to_file.append(buffer_string_conversion);
			for (jatom = 0; jatom < n_spins_in_shell[iatom][0]; ++jatom)
			{
				jspin = neigh[iatom][0][jatom];
				snprintf(buffer_string_conversion, buffer_length, " %12.6f %12.6f %12.6f", neigh_pos[0][iatom][0][jatom], neigh_pos[1][iatom][0][jatom], neigh_pos[2][iatom][0][jatom]);
				output_to_file.append(buffer_string_conversion);
			}//endfor jatom
		}//endfor iatom
		IO::Dump_to_File(output_to_file, "output/neighbours_pos_shell_1.dat");
	}//end Neighbours::Get_Neighbours_in_Shells


	void Neighbours::Create_Neighbours_4Spin(const int nos, const int n_shells, const std::vector<std::vector<std::vector<int>>> & neigh,
		const std::vector<std::vector<int>> &n_spins_in_shell, const int &max_n_4spin, std::vector<int> &n_4spin, std::vector<std::vector<std::vector<int>>> &neigh_4spin)
	{
		//========================= Init local vars ================================
		int shell = 0, triplet;
		int ispin, jspin, kspin, lspin, mspin;
		int jneigh, kneigh, lneigh, mneigh;
		std::vector<double> build_array = { 0.0, 0.0, 0.0 };

		std::string output_to_file = "List of neighbour quadruples in 4-spin interaction\n";

		if (nos < 5E+06) {								// if nos too big - guard preallocation against int overflow
			output_to_file.reserve(nos * 4 * 45 + 100);	// memory preallocation for fast append
		}
		else { output_to_file.reserve(int(1E+08)); }
		const int buffer_length = 45;
		char buffer_string_conversion[buffer_length + 2];
		//------------------------ End Init ----------------------------------------

		for (ispin = 0; ispin < nos; ++ispin) {
			for (jneigh = 0; jneigh < n_spins_in_shell[ispin][shell]; ++jneigh) {
				jspin = neigh[ispin][shell][jneigh];
				for (lneigh = jneigh + 1; lneigh < n_spins_in_shell[ispin][shell]; ++lneigh) {
					lspin = neigh[ispin][shell][lneigh];
					for (kneigh = 0; kneigh < n_spins_in_shell[jspin][shell]; ++kneigh) {
						kspin = neigh[jspin][shell][kneigh];
						for (mneigh = 0; mneigh < n_spins_in_shell[lspin][shell]; ++mneigh) {
							mspin = neigh[lspin][shell][mneigh];
							if (mspin == kspin && kspin != ispin) {
								n_4spin[ispin] = n_4spin[ispin] + 1;
								neigh_4spin[0][ispin][n_4spin[ispin] - 1] = jspin;
								neigh_4spin[1][ispin][n_4spin[ispin] - 1] = kspin;
								neigh_4spin[2][ispin][n_4spin[ispin] - 1] = lspin;
							}//endif
						}//endfor mneigh
					}//enfor kneigh
				}//enfor lneigh
			}//endfor jneigh

			for (triplet = 0; triplet < n_4spin[ispin] - 1; ++triplet) {
				snprintf(buffer_string_conversion, buffer_length, "%7i  %+07i  %+07i  %+07i \n", ispin, neigh_4spin[0][ispin][triplet], neigh_4spin[1][ispin][triplet], neigh_4spin[2][ispin][triplet]);
				output_to_file.append(buffer_string_conversion);
			}//endfor triplet
		}//endfor ispin

		IO::Dump_to_File(output_to_file, "output/neighbours_4spin.dat");
	}//end Neighbours::Create_Neighbours_4Spin

	void Neighbours::Create_DM_Norm_Vectors_Bulk(const int nos, const std::vector<std::vector<double>> &spin_pos, const int number_b_vectors,
		const std::vector<std::vector<double>> &boundary_vectors, const int n_shells, const std::vector<std::vector<int>> &n_spins_in_shell,
		const std::vector<std::vector<std::vector<int>>> & neigh, std::vector<std::vector<std::vector<std::vector<double>>>> & neigh_pos,
		const int max_ndm, std::vector<std::vector<std::vector<double>>> &dm_normal)
	{
		//========================= Init local vars ================================
		int ispin, jspin, jneigh;
		std::vector<double> ispin_pos = { 0.0, 0.0, 0.0 };
		std::vector<double> jspin_pos = { 0.0, 0.0, 0.0 };
		std::vector<double> r_a = { 0.0, 0.0, 0.0 };
		std::vector<double> r_b = { 0.0, 0.0, 1.0 };
		std::vector<double> build_array = { 0.0, 0.0, 0.0 };
		//------------------------ End Init ----------------------------------------

		Log.Send(Log_Level::DEBUG, Log_Sender::ALL, "Calculating Bulk DMI norm vectors...");
		for (ispin = 0; ispin < nos; ++ispin) {								// loop over all spins
			Vectormath::Vector_Copy_o(ispin_pos, spin_pos, 0, ispin);
			for (jneigh = 0; jneigh < n_spins_in_shell[ispin][0]; ++jneigh) {	// loop over all neighbours of that spin
				jspin = neigh[ispin][0][jneigh];
				Vectormath::Vector_Copy_o(jspin_pos, neigh_pos, 0, ispin, 0, jneigh);
				//Vectormath::Array_Array_Add(jspin_pos, ispin_pos, r_a, -1.0, 1.0); //get DMI vec with chirality "-"
				Vectormath::Array_Array_Add(jspin_pos, ispin_pos, r_a, 1.0, -1.0);	// get DMI vec with chirality "+"
				Vectormath::Normalize(r_a);							// normalize DMI vec
				Vectormath::Vector_Copy_i(dm_normal, r_a, 0, ispin, jneigh);
			}//endfor jneigh
		}//endfor ispin
		Log.Send(Log_Level::DEBUG, Log_Sender::ALL, "Done calculating Bulk DMI norm vectors.");
	}//end Neighbours::Create_DM_Norm_Vectors_Bulk

	void Neighbours::Create_DM_Norm_Vectors_Surface(const int nos, const std::vector<std::vector<double>> &spin_pos, const int number_b_vectors,
		const std::vector<std::vector<double>> &boundary_vectors, const int n_shells, const std::vector<std::vector<int>> &n_spins_in_shell,
		const std::vector<std::vector<std::vector<int>>> & neigh, std::vector<std::vector<std::vector<std::vector<double>>>> & neigh_pos,
		const int max_ndm, std::vector<std::vector<std::vector<double>>> &dm_normal)
	{
		//========================= Init local vars ================================
		int ispin, jneigh;
		std::vector<double> unit_vec_z = { 0.0, 0.0, 1.0 };
		std::vector<double> build_array_1 = { 0.0, 0.0, 0.0 };
		std::vector<double> build_array_2 = { 0.0, 0.0, 0.0 };
		//------------------------ End Init ----------------------------------------
		Create_DM_Norm_Vectors_Bulk(nos, spin_pos, number_b_vectors, boundary_vectors, n_shells, n_spins_in_shell, neigh, neigh_pos, max_ndm, dm_normal);

		for (ispin = 0; ispin < nos; ++ispin) {
			for (jneigh = 0; jneigh < max_ndm; ++jneigh) {
				Vectormath::Vector_Copy_o(build_array_1, dm_normal, 0, ispin, jneigh);
				Vectormath::Cross_Product(build_array_1, unit_vec_z, build_array_2);
				Vectormath::Vector_Copy_i(dm_normal, build_array_2, 0, ispin, jneigh);
			}
		}
		Log.Send(Log_Level::DEBUG, Log_Sender::ALL, "Done calculating Surface DMI norm vectors.");
	}//end Neighbours::Create_DM_Norm_Vectors_Surface

	void Neighbours::DM_Norm_Vectors_To_File(const int nos, const int n_shells, const std::vector<std::vector<int>> &n_spins_in_shell,
		const std::vector<std::vector<std::vector<int>>> & neigh, const std::vector<std::vector<std::vector<double>>> &dm_normal)
	{
		//========================= Init local vars ================================
		int ispin, jneigh;
		std::vector<double> build_array = { 0.0, 0.0, 0.0 };
		std::vector<double> unit_vec_z = { 0.0, 0.0, 1.0 };

		std::string output_to_file = "List of neighbours,'\n normalized vector for Dzyaloshinskii-Moriya interaction, and the angle to the normal\n";
		if (nos < 3500) {									// if nos too big - the preallocation gets int overflow
			output_to_file.reserve(nos * nos * 85 + 150);	//memory allocation for fast append
		}
		else { output_to_file.reserve(int(1E+08)); }

		const int buffer_length = 85;
		char buffer_string_conversion[buffer_length + 2];
		//------------------------ End Init ----------------------------------------
		for (ispin = 0; ispin < nos; ++ispin) {
			for (jneigh = 0; jneigh < n_spins_in_shell[ispin][0]; ++jneigh) {
				Vectormath::Vector_Copy_o(build_array, dm_normal, 0, ispin, jneigh);
				snprintf(buffer_string_conversion, buffer_length, "%7i  %7i | %+13.7f  %+13.7f  %+13.7f | %+10.4f \n", ispin, neigh[ispin][0][jneigh], dm_normal[0][ispin][jneigh], dm_normal[1][ispin][jneigh], dm_normal[2][ispin][jneigh], Vectormath::Angle(build_array, unit_vec_z, true));
				output_to_file.append(buffer_string_conversion);
			}//endfor jneigh
		}//endfor ispin
		IO::Dump_to_File(output_to_file, "output/neighbours_DM.dat");
	}//end Neighbours::DM_Norm_Vectors_To_File

	void Neighbours::Create_Segments(const Data::Geometry & geometry, const int n_shells,
		const int nos, const std::vector<std::vector<double>> &spin_pos, const std::vector<std::vector<int>> &n_spins_in_shell,
		const std::vector<std::vector<std::vector<int>>> & neigh, std::vector<std::vector<std::vector<std::vector<double>>>> & neigh_pos,
		std::vector<std::vector<int>> &segments, std::vector<std::vector<std::vector<double>>> &segments_pos)
	{
		//========================= Init local vars ================================
		// Retrieve basis vectors from geometry.basis
		std::vector<double> a = { geometry.basis[0][0], geometry.basis[1][0], geometry.basis[2][0] };
		std::vector<double> b = { geometry.basis[0][1], geometry.basis[1][1], geometry.basis[2][1] };
		std::vector<double> c = { geometry.basis[0][2], geometry.basis[1][2], geometry.basis[2][2] };

		std::vector<double> ipos = { 0.0, 0.0, 0.0 };
		std::vector<double> jpos = { 0.0, 0.0, 0.0 };
		double error = 1.0E-5;
		int ispin, jneigh, jspin, shell, shell_fin;
		std::vector<double> build_array = { 0.0, 0.0, 0.0 };

		std::string output_to_file = "List of spins in segments\n";
		const int buffer_length = 45;
		output_to_file.reserve(n_shells * buffer_length);	//memory allocation for fast append
		char buffer_string_conversion[buffer_length];
		//------------------------ End Init ----------------------------------------

		if (n_shells >= 2) {
			shell_fin = 2;
		}
		else {
			shell_fin = 1;
		}

		for (ispin = 0; ispin < nos; ++ispin) {
			ipos[0] = spin_pos[0][ispin]; ipos[1] = spin_pos[1][ispin]; ipos[2] = spin_pos[2][ispin];
			for (shell = 0; shell < shell_fin; ++shell) {
				for (jneigh = 0; jneigh < n_spins_in_shell[ispin][shell]; ++jneigh) {
					jspin = neigh[ispin][shell][jneigh];
					//jpos[0] = spin_pos[0][jspin]; jpos[1] = spin_pos[1][jspin]; jpos[2] = spin_pos[2][jspin];
					jpos[0] = neigh_pos[0][ispin][shell][jneigh]; jpos[1] = neigh_pos[1][ispin][shell][jneigh];	jpos[2] = neigh_pos[2][ispin][shell][jneigh];

					Vectormath::Array_Array_Add(ipos, a, jpos, build_array, 1.0, 1.0, -1.0);
					if (Vectormath::Length(build_array) < error) {
						segments[ispin][0] = jspin;
						segments_pos[0][ispin][0] = jpos[0]; segments_pos[1][ispin][0] = jpos[1]; segments_pos[2][ispin][0] = jpos[2];
					}
					Vectormath::Array_Array_Add(ipos, b, jpos, build_array, 1.0, 1.0, -1.0);
					if (Vectormath::Length(build_array) < error) {
						segments[ispin][1] = jspin;
						segments_pos[0][ispin][1] = jpos[0]; segments_pos[1][ispin][1] = jpos[1]; segments_pos[2][ispin][1] = jpos[2];
					}
					Vectormath::Array_Array_Add(ipos, a, jpos, build_array, 1.0, -1.0, -1.0);
					if (Vectormath::Length(build_array) < error) {
						segments[ispin][2] = jspin;
						segments_pos[0][ispin][2] = jpos[0]; segments_pos[1][ispin][2] = jpos[1]; segments_pos[2][ispin][2] = jpos[2];
					}
					Vectormath::Array_Array_Add(ipos, b, jpos, build_array, 1.0, -1.0, -1.0);
					if (Vectormath::Length(build_array) < error) {
						segments[ispin][3] = jspin;
						segments_pos[0][ispin][3] = jpos[0]; segments_pos[1][ispin][3] = jpos[1]; segments_pos[2][ispin][3] = jpos[2];
					}
				}//endfor jneigh
			}//endfor shell
			snprintf(buffer_string_conversion, buffer_length, "%+06i  %+06i  %+06i  %+06i  %+06i\n", ispin, segments[ispin][0], segments[ispin][1], segments[ispin][2], segments[ispin][3]);
			output_to_file.append(buffer_string_conversion);
		}//endfor ispin
		IO::Dump_to_File(output_to_file, "output/segments.dat");
	}//end Neighbours::Create_Segments

	void Neighbours::Create_Dipole_Pairs(const Data::Geometry & geometry, double dd_radius,
		std::vector<std::vector<std::vector<int>>> & DD_indices, std::vector<std::vector<double>> & DD_magnitude, std::vector<std::vector<std::vector<double>>> & DD_normal)
	{
		// Get the boundary vectors
		// auto boundary_vectors = Engine::Neighbours::Get_Boundary_Vectors(geometry, std::vector<bool>{ true, true, true });
		
		// ------ Find the pairs for the first cell ------
		auto vector_ij = std::vector<double>(3), build_array = std::vector<double>(3),  ipos = std::vector<double>(3), jpos = std::vector<double>(3);
		double magnitude;
		
		int iatom, jatom;
		int da, db, dc;
		int sign_a, sign_b, sign_c;
		int pair_da, pair_db, pair_dc;
		int na, nb, nc;

		unsigned int bvector;

		int idx_i = 0, idx_j = 0;
		int Na = geometry.n_cells[0];
		int Nb = geometry.n_cells[1];
		int Nc = geometry.n_cells[2];
		int N = geometry.n_spins_basic_domain;
		
		int periods_a, periods_b, periods_c, pair_periodicity;

		// Loop over all basis atoms
		for (iatom = 0; iatom < N; ++iatom)
		{
			for (jatom = 0; jatom < N; ++jatom)
			{
				// Because the terms with the largest distance are the smallest, we start with the largest indices
				for (da = Na-1; da >= 0; --da)
				{
					for (db = Nb-1; db >= 0; --db)
					{
						for (dc = Nc-1; dc >= 0; --dc)
						{
							for (sign_a = -1; sign_a <= 1; sign_a+=2)
							{
							for (sign_b = -1; sign_b <= 1; sign_b+=2)
							{
							for (sign_c = -1; sign_c <= 1; sign_c+=2)
							{
								pair_da = sign_a * da;
								pair_db = sign_b * db;
								pair_dc = sign_c * dc;
								// Calculate positions and difference vector
								for (int dim=0; dim<3; ++dim)
								{
									ipos[dim] 		= geometry.spin_pos[dim][iatom];
									jpos[dim] 		= geometry.spin_pos[dim][jatom]
														+ geometry.translation_vectors[dim][0]*pair_da
														+ geometry.translation_vectors[dim][1]*pair_db
														+ geometry.translation_vectors[dim][2]*pair_dc;
									vector_ij[dim]  = jpos[dim] - ipos[dim];
								}
								// Length of difference vector
								magnitude = Vectormath::Length(vector_ij);
								if ( magnitude==0.0 || (da==0 && sign_a==-1) || (db==0 && sign_b==-1) || (dc==0 && sign_c==-1) )
								{
									magnitude = dd_radius + 1.0;
								}
								// Check if inside DD radius
								if ( magnitude - dd_radius < 1.0E-5 )
								{
									// std::cerr << "found " << iatom << " " << jatom << std::endl;
									// std::cerr << "      " << pair_da << " " << pair_db << " " << pair_dc << std::endl;
									// Normal
									vector_ij[0] = vector_ij[0]/magnitude;
									vector_ij[1] = vector_ij[1]/magnitude;
									vector_ij[2] = vector_ij[2]/magnitude;

									// ------ Translate for the whole lattice ------
									// Create all Pairs of this Kind through translation
									for (int na = 0; na < Na; ++na)
									{
										for (int nb = 0; nb < Nb; ++nb)
										{
											for (int nc = 0; nc < Nc; ++nc)
											{
												idx_i = iatom + N*na + N*Na*nb + N*Na*Nb*nc;
												// na + pair_da is absolute position of cell in x direction
												// if (na + pair_da) > Na (number of atoms in x)
												// go to the other side with % Na
												// if (na + pair_da) negative (or multiply (of Na) negative)
												// add Na and modulo again afterwards
												// analogous for y and z direction with nb, nc
												idx_j = jatom	+ N*( (((na + pair_da) % Na) + Na) % Na )
																+ N*Na*( (((nb + pair_db) % Nb) + Nb) % Nb )
																+ N*Na*Nb*( (((nc + pair_dc) % Nc) + Nc) % Nc );
												// Determine the periodicity
												periods_a = (na + pair_da) / Na;
												periods_b = (nb + pair_db) / Nb;
												periods_c = (nc + pair_dc) / N;
												//		none
												if (periods_a == 0 && periods_b == 0 && periods_c == 0)
												{
													pair_periodicity = 0;
												}
												//		a
												else if (periods_a != 0 && periods_b == 0 && periods_c == 0)
												{
													pair_periodicity = 1;
												}
												//		b
												else if (periods_a == 0 && periods_b != 0 && periods_c == 0)
												{
													pair_periodicity = 2;
												}
												//		c
												else if (periods_a == 0 && periods_b == 0 && periods_c != 0)
												{
													pair_periodicity = 3;
												}
												//		ab
												else if (periods_a != 0 && periods_b != 0 && periods_c == 0)
												{
													pair_periodicity = 4;
												}
												//		ac
												else if (periods_a != 0 && periods_b == 0 && periods_c != 0)
												{
													pair_periodicity = 5;
												}
												//		bc
												else if (periods_a == 0 && periods_b != 0 && periods_c != 0)
												{
													pair_periodicity = 6;
												}
												//		abc
												else if (periods_a != 0 && periods_b != 0 && periods_c != 0)
												{
													pair_periodicity = 7;
												}

												// Add the indices and parameters to the corresponding lists
												if (idx_i < idx_j)
												{
													// std::cerr << "   made pair " << idx_i << " " << idx_j << std::endl;
													DD_indices[pair_periodicity].push_back(std::vector<int>{ idx_i, idx_j });
													DD_magnitude[pair_periodicity].push_back(magnitude);
													DD_normal[pair_periodicity].push_back(vector_ij);
												}
											}// end for nc
										}// end for nb
									}// end for na
								}// end if in radius
							}
							}
							}
							// else 
							// 	std::cerr << "outed " << iatom << " " << jatom << " with d=" << magnitude << std::endl;
						}// end for pair_dc
					}// end for pair_db
				}// end for pair_da
			}// end for jatom
		}// end for iatom
	}


	void Neighbours::Create_Dipole_Neighbours(const Data::Geometry & geometry, std::vector<bool> boundary_conditions,
		const double dd_radius,	std::vector<std::vector<int>>& dd_neigh, std::vector<std::vector<std::vector<double>>>& dd_neigh_pos,
		std::vector<std::vector<std::vector<double>>>& dd_normal, std::vector<std::vector<double>> & dd_distance)
	{
		std::vector<int> max_n_array;
		auto shell_radius = std::vector<double>(1, dd_radius);
		std::vector<std::vector<double>> boundary_vectors;

		// Calculate boundary vectors
		boundary_vectors = Get_Boundary_Vectors(geometry, boundary_conditions);

		// Get maximum number of neighbours in the DD radius
		max_n_array = Get_MaxNumber_NInShell(geometry, 1, shell_radius, boundary_vectors, false);
		int max_n = max_n_array[0];
		
		//------------- Allocate output vectors ---------------
		// Dipole Dipole neighbours of each spin neigh_dd[nos][max_n]
		dd_neigh = std::vector<std::vector<int>>(geometry.nos, std::vector<int>(max_n));
		// Dipole Dipole neighbour positions [dim][nos][max_n]
		// dd_neigh_pos = std::vector<std::vector<std::vector<std::vector<double>>>>(3, std::vector<std::vector<std::vector<double>>>(nos, std::vector<std::vector<double>>(n_shells, std::vector<double>(max_number_n))));
		dd_neigh_pos = std::vector<std::vector<std::vector<double>>>(3, std::vector<std::vector<double>>(geometry.nos, std::vector<double>(max_n)));
		// Dipole Dipole normal vectors [dim][nos][1][max_n]
		dd_normal = std::vector<std::vector<std::vector<double>>>(3, std::vector<std::vector<double>>(geometry.nos, std::vector<double>(max_n)));
		// Dipole Dipole normal vectors [dim][nos][1][max_n]
		dd_distance = std::vector<std::vector<double>>(geometry.nos, std::vector<double>(max_n));

		if (dd_radius > 0.0)
		{
			// ==== Use this stuff to be able to use Get_Neighbours_in_Shells for 1 single shell ====
			auto neigh = std::vector<std::vector<std::vector<int>>>(geometry.nos, std::vector<std::vector<int>>(1, std::vector<int>(max_n)));
			auto neigh_pos = std::vector<std::vector<std::vector<std::vector<double>>>>(3, std::vector<std::vector<std::vector<double>>>(geometry.nos, std::vector<std::vector<double>>(1, std::vector<double>(max_n, 0))));
			auto n_spins_in_shell = std::vector<std::vector<int>>(geometry.nos, std::vector<int>(1));

			Get_Neighbours_in_Shells(geometry, 1, shell_radius, boundary_vectors, n_spins_in_shell, neigh, neigh_pos, false);

			int ispin, jneigh, dim;
			double length = 0.0;
			for (ispin = 0; ispin < geometry.nos; ++ispin) {
				// transfer number of neighbours into n_dd_neigh
				for (jneigh = 0; jneigh < n_spins_in_shell[ispin][0]; ++jneigh) {
					// transfer neighbours back to dd_neigh vector
					dd_neigh[ispin][jneigh] = neigh[ispin][0][jneigh];
					for (dim = 0; dim < 3; ++dim) {
						dd_neigh_pos[dim][ispin][jneigh] = neigh_pos[dim][ispin][0][jneigh];
						// calculate dd_normal
						dd_normal[dim][ispin][jneigh] = geometry.spin_pos[dim][ispin] - dd_neigh_pos[dim][ispin][jneigh];
					}// endfor dim
					dd_distance[ispin][jneigh] = Vectormath::Length(dd_normal, ispin, jneigh);
					if (dd_distance[ispin][jneigh] == 0.0) throw Exception::Division_by_zero;
					for (dim = 0; dim < 3; ++dim) {
						dd_normal[dim][ispin][jneigh] = dd_normal[dim][ispin][jneigh] / dd_distance[ispin][jneigh];
					}
				}// endfor jneigh
			}// endfor ispin
		}
	}// end Neighbours::Create_Dipole_Neighbours

	std::vector<std::vector<double>> Neighbours::Get_Boundary_Vectors(const Data::Geometry & geometry, const std::vector<bool> & boundary_conditions)
	{
		// Determine vector size
		int n_boundaries=0, n_boundary_vectors=0;
		for (int i = 0; i < 3; ++i) if (boundary_conditions[i]) n_boundaries++;
		n_boundary_vectors = (int)std::pow(3, n_boundaries);
		auto boundary_vectors = std::vector<std::vector<double>>(3, std::vector<double>(n_boundary_vectors, 0.0));
		
		// Determine the runs we take over the basis directions
		std::vector<int> list_ia(1, 0), list_ib(1, 0), list_ic(1, 0);
		if (boundary_conditions[0]) {
			for (int ia = -1; ia < +2; ++ia) {
				if (ia != 0) list_ia.push_back(ia);
			}
		}
		if (boundary_conditions[1]) {
			for (int ib = -1; ib < +2; ++ib) {
				if (ib != 0) list_ib.push_back(ib);
			}
		}
		if (boundary_conditions[2]) {
			for (int ic = -1; ic < +2; ++ic) {
				if (ic != 0) list_ic.push_back(ic);
			}
		}

		// Fill the vector
		for (int dim = 0; dim < 3; ++dim)
		{
			int temp = 1;
			for (int ia : list_ia)
			{
				for (int ib : list_ib)
				{
					for (int ic : list_ic)
					{
						if (!(ia == 0 && ib == 0 && ic == 0))
						{
							boundary_vectors[dim][temp] = geometry.translation_vectors[dim][0] * geometry.n_cells[0] * ia
														+ geometry.translation_vectors[dim][1] * geometry.n_cells[1] * ib
														+ geometry.translation_vectors[dim][2] * geometry.n_cells[2] * ic;
							temp++;
						}
					}
				}
			}
		}
		return boundary_vectors;
	}// END Get_Boundary_Vectors

	void Neighbours::Create_DD_Pairs_from_Neighbours(const Data::Geometry & geometry, const std::vector<std::vector<int>> & dd_neighbours, const std::vector<std::vector<std::vector<double>>> & dd_neighbours_positions, const std::vector<std::vector<double>> & dd_distance, const std::vector<std::vector<std::vector<double>>> & dd_normal,
		std::vector<std::vector<std::vector<int>>> & DD_indices, std::vector<std::vector<double>> & DD_magnitude, std::vector<std::vector<std::vector<double>>> & DD_normal)
	{
		int i_spin = 0, i_neigh = 0;

		int periodicity;
		double dot_a, dot_b, dot_c;
		double norm_ta=0, norm_tb=0, norm_tc=0;
		double norm_ba=0, norm_bb=0, norm_bc=0;
		bool periodical_a, periodical_b, periodical_c;

		Log.Send(Log_Level::WARNING, Log_Sender::ALL, "DD Pairs are being calculated from Neighbours! This may produce bugs in non-orthogonal basis systems!");
		
		// Norm of basis and translation vectors
		for (int dim = 0; dim < 3; ++dim)
		{
			norm_ba += pow(geometry.basis[dim][0], 2.0);
			norm_bb += pow(geometry.basis[dim][1], 2.0);
			norm_bc += pow(geometry.basis[dim][2], 2.0);
			
			norm_ta += pow(geometry.translation_vectors[dim][0], 2.0);
			norm_tb += pow(geometry.translation_vectors[dim][1], 2.0);
			norm_tc += pow(geometry.translation_vectors[dim][2], 2.0);
		}
		norm_ba = sqrt(norm_ba); norm_bb = sqrt(norm_bb); norm_bc = sqrt(norm_bc);
		norm_ta = sqrt(norm_ta); norm_tb = sqrt(norm_tb); norm_tc = sqrt(norm_tc);
		// ...
		for (auto shell : dd_neighbours)
		{
			i_neigh = 0;
			for (auto neigh : shell)
			{
				periodicity = 0;
				periodical_a = true; periodical_b = true; periodical_c = true;
				dot_a = 0; dot_b = 0; dot_c = 0;
				// Dot products with basis vectors to determine if outside of boundaries
				for (int dim = 0; dim < 3; ++dim)
				{
						dot_a += (dd_neighbours_positions[dim][i_spin][i_neigh]) * geometry.translation_vectors[dim][0] / norm_ta / norm_ba;
						dot_b += (dd_neighbours_positions[dim][i_spin][i_neigh]) * geometry.translation_vectors[dim][1] / norm_tb / norm_bb;
						dot_c += (dd_neighbours_positions[dim][i_spin][i_neigh]) * geometry.translation_vectors[dim][2] / norm_tc / norm_bc;
				}
				// Determine periodical boundaries
				if (0 <= dot_a && dot_a < geometry.n_cells[0]) periodical_a = false;
				else periodical_a = true;
				if (0 <= dot_b && dot_b < geometry.n_cells[1]) periodical_b = false;
				else periodical_b = true;
				if (0 <= dot_c && dot_c < geometry.n_cells[2]) periodical_c = false;
				else periodical_c = true;
				// Determine periodicity (array index)
				//		none
				if (!periodical_a && !periodical_b && !periodical_c)		periodicity = 0;
				//		a
				else if (periodical_a && !periodical_b && !periodical_c)	periodicity = 1;
				//		b
				else if (!periodical_a && periodical_b && !periodical_c)	periodicity = 2;
				//		c
				else if (!periodical_a && !periodical_b && periodical_c)	periodicity = 3;
				//		ab
				else if (periodical_a && periodical_b && !periodical_c)		periodicity = 4;
				//		ac
				else if (periodical_a && !periodical_b && periodical_c)		periodicity = 5;
				//		bc
				else if (!periodical_a && periodical_b && periodical_c)		periodicity = 6;
				//		abc
				else if (periodical_a && periodical_b && periodical_c)		periodicity = 7;

				// Remove double counting by only taking pairs with i<j
				if (i_spin < neigh && dd_distance[i_spin][i_neigh] != 0)
				{
					DD_indices[periodicity].push_back({ i_spin, neigh });
					DD_magnitude[periodicity].push_back(dd_distance[i_spin][i_neigh]);
					DD_normal[periodicity].push_back({ dd_normal[0][i_spin][i_neigh], dd_normal[1][i_spin][i_neigh], dd_normal[2][i_spin][i_neigh] });
				}
				++i_neigh;
			}
			++i_spin;
		}
	}

}// end Namespace Engine