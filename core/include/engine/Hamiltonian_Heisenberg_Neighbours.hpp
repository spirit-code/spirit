#pragma once
#ifndef HAMILTONIAN_HEISENBERG_NEIGHBOURS_H
#define HAMILTONIAN_HEISENBERG_NEIGHBOURS_H

#include <vector>

#include "Spirit_Defines.h"
#include <engine/Vectormath_Defines.hpp>
#include <engine/Hamiltonian.hpp>
#include <data/Geometry.hpp>

namespace Engine
{
	// Hamiltonian contains all exchange- and interaction-information about the spin system
	class Hamiltonian_Heisenberg_Neighbours : public Hamiltonian
	{
	public:
		// Constructor
		Hamiltonian_Heisenberg_Neighbours(std::vector<bool> boundary_conditions, scalar external_field_magnitude, Vector3 external_field_normal, scalar mu_s,
			scalar anisotropy_magnitude, Vector3 anisotropy_normal,
			int n_neigh_shells, std::vector<scalar> jij, scalar dij, int dm_chirality, scalar bij, scalar kijkl, scalar dd_radius, Data::Geometry geometry);
		
		void Update_Energy_Contributions() override;
		
		void Hessian(const vectorfield & spins, MatrixX & hessian) override;
		void Gradient(const vectorfield & spins, vectorfield & gradient) override;
		void Energy_Contributions_per_Spin(const vectorfield & spins, std::vector<std::pair<std::string, scalarfield>> & contributions) override;

		// Hamiltonian name as string
		const std::string& Name() override;

		// -------------------- Single Spin Interactions ------------------
		// External Magnetic Field
		scalar external_field_magnitude;
		Vector3 external_field_normal;
		scalar mu_s;
		// Anisotropy
		scalar anisotropy_magnitude;
		Vector3 anisotropy_normal;

		// -------------------- Two Spin Interactions ------------------
		// number of pairwise interaction shells
		int n_neigh_shells;
		// number of spins in shell ns_in_shell[
		std::vector<std::vector<int>> n_spins_in_shell;
		// Neighbours of each spin neigh[nos][n_shell][max_n_in_shell[n_shell]]
		std::vector<std::vector<std::vector<int>>> neigh;

		// Exchange Interaction
		std::vector<scalar> jij;
		// DMI
		scalar dij;
		// DM normal vectors [nos][max_n_in_shell[n_shell]]
		std::vector<vectorfield> dm_normal;
		// Biquadratic Exchange
		scalar bij;
		// Dipole Dipole radius
		scalar dd_radius;
		// Dipole Dipole neighbours of each spin neigh_dd[nos][max_n]
		std::vector<std::vector<int>> dd_neigh;
		// Dipole Dipole neighbour positions of each spin neigh_dd[nos][max_n]
		std::vector<std::vector<Vector3>> dd_neigh_pos;
		// Dipole Dipole normal vectors [nos][max_n]
		std::vector<vectorfield> dd_normal;
		// Dipole Dipole distance [nos][max_n]
		std::vector<std::vector<scalar>> dd_distance;

		// -------------------- Four Spin Interactions ------------------
		// Four Spin
		scalar kijkl;
		// Maximum number of 4-Spin interactions per spin
		int max_n_4spin;
		// actual number of 4-Spin interactions for each spin number_t4_spin[nos]
		std::vector<int> n_4spin;
		// 4 spin interaction neighbours: neigh_4spin[dim][nos][number_t4_spin[nos]]
		std::vector<std::vector<std::vector<int>>> neigh_4spin;

		// segments[nos][4]
		std::vector<std::vector<int>> segments;
		// Position of the Segments: segments_pos[nos][4]
		std::vector<std::vector<Vector3>> segments_pos;

	private:
		// -------------------- Effective Field Functions ------------------
		// Calculates the Zeeman contribution to the effective field of spin ispin within system s
		void Field_Zeeman(int nos, const vectorfield & spins, vectorfield & eff_field, const int ispin);
		// Calculates the Exchange contribution to the effective field of spin ispin within system s
		void Field_Exchange(int nos, const vectorfield & spins, vectorfield & eff_field, const int ispin);
		// Calculates the Anisotropic contribution to the effective field of spin ispin within system s
		void Field_Anisotropic(int nos, const vectorfield & spins, vectorfield & eff_field, const int ispin);
		// Calculates the Biquadratic Coupling contribution to the effective field of spin ispin within system s
		void Field_BQC(int nos, const vectorfield & spins, vectorfield & eff_field, const int ispin);
		// Calculates the 4-spin Coupling contribution to the effective field of spin ispin within system s
		void Field_FourSC(int nos, const vectorfield & spins, vectorfield & eff_field, const int ispin);
		// Calculates the Dzyaloshinskii-Moriya Interaction contribution to the effective field of spin ispin within system s
		void Field_DM(int nos, const vectorfield & spins, vectorfield & eff_field, const int ispin);
		// Calculates the Dipole-Dipole contribution to the effective field of spin ispin within system s
		void Field_DipoleDipole(int nos, const vectorfield & spins, vectorfield & eff_field, const int ispin);

		// -------------------- Energy Functions ------------------
		// Indices for Energy vector
		int idx_zeeman, idx_anisotropy, idx_exchange, idx_dmi, idx_bqc, idx_fsc, idx_dd;
		// calculates the Zeeman Energy of spin ispin within system s
		void E_Zeeman(const vectorfield & spins, scalarfield & Energy);
		// calculates the Exchange Energy of spin ispin within system s
		void E_Exchange(const vectorfield & spins, scalarfield & Energy);
		// calculates the Anisotropic Energy of spin ispin within system s
		void E_Anisotropic(const vectorfield & spins, scalarfield & Energy);
		// calculates the Biquadratic Coupling Energy of spin ispin within system s
		void E_BQC(const vectorfield & spins, scalarfield & Energy);
		// calculates the 4-spin Coupling Energy of spin ispin within system s
		void E_FourSC(const vectorfield & spins, scalarfield & Energy);
		// calculates the Dzyaloshinskii-Moriya Interaction Energy of spin ispin within system s
		void E_DM(const vectorfield & spins, scalarfield & Energy);
		// calculates the Dipole-Dipole Energy of spin ispin within system s
		void E_DipoleDipole(const vectorfield & spins, scalarfield & Energy);
	};
}
#endif