#pragma once
#ifndef HAMILTONIAN_ISOTROPIC_NEW_H
#define HAMILTONIAN_ISOTROPIC_NEW_H

#include <vector>

#include "Hamiltonian.h"
#include "Geometry.h"

namespace Engine
{
	// Hamiltonian contains all exchange- and interaction-information about the spin system
	class Hamiltonian_Isotropic : public Hamiltonian
	{
	public:
		// Constructor
		Hamiltonian_Isotropic(std::vector<bool> boundary_conditions, double external_field_magnitude, std::vector<double> external_field_normal, double mu_s,
			double anisotropy_magnitude, std::vector<double> anisotropy_normal,
			int n_neigh_shells, std::vector<double> jij, double dij, double bij, double kijkl, double dd_radius, Data::Geometry geometry);

		void Effective_Field(std::vector<double> & spins, std::vector<double> & field) override;
		double Energy(std::vector<double> & spins) override;
		std::vector<double> Energy_Array(std::vector<double> & spins) override;
		// Need to implement:
		//std::vector<std::vector<double>> Energy_Array_per_Spin(std::vector<double> & spins) override;

	//private:// these are currently needed by the to-be-removed Energy and Eff_Field
		// -------------------- Effective Field Functions ------------------
		// Calculates the Zeeman contribution to the effective field of spin ispin within system s
		void Field_Zeeman(int nos, std::vector<double> & spins, std::vector<double> & eff_field, const int ispin);
		// Calculates the Exchange contribution to the effective field of spin ispin within system s
		void Field_Exchange(int nos, std::vector<double> & spins, std::vector<double> & eff_field, const int ispin);
		// Calculates the Anisotropic contribution to the effective field of spin ispin within system s
		void Field_Anisotropic(int nos, std::vector<double> & spins, std::vector<double> & eff_field, const int ispin);
		// Calculates the Biquadratic Coupling contribution to the effective field of spin ispin within system s
		void Field_BQC(int nos, std::vector<double> & spins, std::vector<double> & eff_field, const int ispin);
		// Calculates the 4-spin Coupling contribution to the effective field of spin ispin within system s
		void Field_FourSC(int nos, std::vector<double> & spins, std::vector<double> & eff_field, const int ispin);
		// Calculates the Dzyaloshinskii-Moriya Interaction contribution to the effective field of spin ispin within system s
		void Field_DM(int nos, std::vector<double> & spins, std::vector<double> & eff_field, const int ispin);
		// Calculates the Dipole-Dipole contribution to the effective field of spin ispin within system s
		void Field_DipoleDipole(int nos, std::vector<double> & spins, std::vector<double> & eff_field, const int ispin);

		// -------------------- Energy Functions ------------------
		// calculates the Zeeman Energy of spin ispin within system s
		double E_Zeeman(int nos, std::vector<double> & spins, const int ispin);
		// calculates the Exchange Energy of spin ispin within system s
		double E_Exchange(int nos, std::vector<double> & spins, const int ispin);
		// calculates the Anisotropic Energy of spin ispin within system s
		double E_Anisotropic(int nos, std::vector<double> & spins, const int ispin);
		// calculates the Biquadratic Coupling Energy of spin ispin within system s
		double E_BQC(int nos, std::vector<double> & spins, const int ispin);
		// calculates the 4-spin Coupling Energy of spin ispin within system s
		double E_FourSC(int nos, std::vector<double> & spins, const int ispin);
		// calculates the Dzyaloshinskii-Moriya Interaction Energy of spin ispin within system s
		double E_DM(int nos, std::vector<double> & spins, const int ispin);
		// calculates the Dipole-Dipole Energy of spin ispin within system s
		double E_DipoleDipole(int nos, std::vector<double> & spins, const int ispin);


		// -------------------- Single Spin Interactions ------------------
		// External Magnetic Field
		double external_field_magnitude;
		std::vector<double> external_field_normal;
		double mu_s;
		// Anisotropy
		double anisotropy_magnitude;
		std::vector<double> anisotropy_normal;

		// -------------------- Two Spin Interactions ------------------
		// number of pairwise interaction shells
		int n_neigh_shells;
		// number of spins in shell ns_in_shell[
		std::vector<std::vector<int>> n_spins_in_shell;
		// Neighbours of each spin neigh[nos][n_shell][max_n_in_shell[n_shell]]
		std::vector<std::vector<std::vector<int>>> neigh;

		// Exchange Interaction
		std::vector<double> jij;
		// DMI
		double dij;
		// DM normal vectors [dim][nos][max_n_in_shell[n_shell]]
		std::vector<std::vector<std::vector<double>>> dm_normal;
		// Biquadratic Exchange
		double bij;
		// Dipole Dipole radius
		double dd_radius;
		// Dipole Dipole neighbours of each spin neigh_dd[nos][max_n]
		std::vector<std::vector<int>> dd_neigh;
		// Dipole Dipole neighbour positions of each spin neigh_dd[dim][nos][max_n]
		std::vector<std::vector<std::vector<double>>> dd_neigh_pos;
		// Dipole Dipole normal vectors [dim][nos][max_n]
		std::vector<std::vector<std::vector<double>>> dd_normal;
		// Dipole Dipole distance [nos][max_n]
		std::vector<std::vector<double>> dd_distance;
		
		// -------------------- Four Spin Interactions ------------------
		// Four Spin
		double kijkl;
		// Maximum number of 4-Spin interactions per spin
		int max_n_4spin;
		// actual number of 4-Spin interactions for each spin number_t4_spin[nos]
		std::vector<int> n_4spin;
		// 4 spin interaction neighbours: neigh_4spin[dim][nos][number_t4_spin[nos]]
		std::vector<std::vector<std::vector<int>>> neigh_4spin;

		// segments[nos][4]
		std::vector<std::vector<int>> segments;
		// Position of the Segments: segments_pos[dim][nos][4]
		std::vector<std::vector<std::vector<double>>> segments_pos;
	};
}
#endif