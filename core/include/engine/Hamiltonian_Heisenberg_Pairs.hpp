#pragma once
#ifndef HAMILTONIAN_HEISENBERG_PAIRS_H
#define HAMILTONIAN_HEISENBERG_PAIRS_H

#include <vector>
#include <memory>

#include "Spirit_Defines.h"
#include <engine/Vectormath_Defines.hpp>
#include <engine/Hamiltonian.hpp>
#include <data/Geometry.hpp>

namespace Engine
{
	/*
		The Heisenberg Hamiltonian using Pairs contains all information on the interactions between spins.
		The information is presented in pair lists and parameter lists in order to easily e.g. calculate the energy of the system via summation.
		Calculations are made on a per-pair basis running over all pairs.
	*/
	class Hamiltonian_Heisenberg_Pairs : public Hamiltonian
	{
	public:
		// Constructor
		Hamiltonian_Heisenberg_Pairs(
			scalarfield mu_s,
			intfield external_field_index, scalarfield external_field_magnitude, vectorfield external_field_normal,
			intfield anisotropy_index, scalarfield anisotropy_magnitude, vectorfield anisotropy_normal,
			pairfield exchange_pair, scalarfield exchange_magnitude,
			pairfield dmi_pair, scalarfield dmi_magnitude, vectorfield dmi_normal,
			scalar ddi_radius,
			quadrupletfield quadruplet, scalarfield quadruplet_magnitude,
			std::shared_ptr<Data::Geometry> geometry,
			intfield boundary_conditions
		);

		void Update_Energy_Contributions() override;

		void Hessian(const vectorfield & spins, MatrixX & hessian) override;
		void Gradient(const vectorfield & spins, vectorfield & gradient) override;
		void Energy_Contributions_per_Spin(const vectorfield & spins, std::vector<std::pair<std::string, scalarfield>> & contributions) override;

		// Hamiltonian name as string
		const std::string& Name() override;

		// ------------ General Variables ------------
		//std::vector<scalar> & field;
		
		// ------------ Single Spin Interactions ------------
		// Spin moment
		scalarfield mu_s;									// [nos]
		// External Magnetic Field
		intfield external_field_indices;
		scalarfield external_field_magnitudes;	// [nos]
		vectorfield external_field_normals;		// [nos] (x, y, z)
		// Anisotropy
		intfield anisotropy_indices;
		scalarfield anisotropy_magnitudes;		// [nos]
		vectorfield anisotropy_normals;			// [nos] (x, y, z)

		// ------------ Pair Interactions ------------
		// Exchange interaction
		pairfield   exchange_pairs;		// [periodicity][nop][2] (i,j)
		scalarfield exchange_magnitudes;	// [periodicity][nop]    J_ij
		// DMI
		pairfield   dmi_pairs;			// [periodicity][nop][2] (i,j)
		scalarfield dmi_magnitudes;			// [periodicity][nop]    D_ij
		vectorfield dmi_normals;			// [periodicity][nop][3] (Dx,Dy,Dz)
		// Dipole Dipole interaction
		pairfield ddi_pairs;			// [periodicity][nop][2] (i,j)
		scalarfield ddi_magnitudes;			// [periodicity][nop]    r_ij (distance)
		vectorfield ddi_normals;			// [periodicity][nop][4] (nx,ny,nz)

		// ------------ Quadruplet Interactions ------------
		quadrupletfield quadruplets;
		scalarfield     quadruplet_magnitudes;

	private:
		std::shared_ptr<Data::Geometry> geometry;

		// ------------ Effective Field Functions ------------
		// Calculate the Zeeman effective field of a single Spin
		void Gradient_Zeeman(vectorfield & gradient);
		// Calculate the Anisotropy effective field of a single Spin
		void Gradient_Anisotropy(const vectorfield & spins, vectorfield & gradient);
		// Calculate the exchange interaction effective field of a Spin Pair
		void Gradient_Exchange(const vectorfield & spins, vectorfield & gradient);
		// Calculate the DMI effective field of a Spin Pair
		void Gradient_DMI(const vectorfield & spins, vectorfield & gradient);
		// Calculates the Dipole-Dipole contribution to the effective field of spin ispin within system s
		void Gradient_DDI(const vectorfield& spins, vectorfield & gradient);
		// Quadruplet
		void Gradient_Quadruplet(const vectorfield & spins, vectorfield & gradient);

		// ------------ Energy Functions ------------
		// Indices for Energy vector
		int idx_zeeman, idx_anisotropy, idx_exchange, idx_dmi, idx_dd, idx_quadruplet;
		// Calculate the Zeeman energy of a Spin System
		void E_Zeeman(const vectorfield & spins, scalarfield & Energy);
		// Calculate the Anisotropy energy of a Spin System
		void E_Anisotropy(const vectorfield & spins, scalarfield & Energy);
		// Calculate the exchange interaction energy of a Spin System
		void E_Exchange(const vectorfield & spins, scalarfield & Energy);
		// Calculate the DMI energy of a Spin System
		void E_DMI(const vectorfield & spins, scalarfield & Energy);
		// calculates the Dipole-Dipole Energy
		void E_DDI(const vectorfield& spins, scalarfield & Energy);
		// Quadruplet
		void E_Quadruplet(const vectorfield & spins, scalarfield & Energy);

	};
}
#endif