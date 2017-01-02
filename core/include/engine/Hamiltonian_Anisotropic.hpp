#pragma once
#ifndef HAMILTONIAN_ANISOTROPIC_NEW_H
#define HAMILTONIAN_ANISOTROPIC_NEW_H

#include <vector>

#include "Core_Defines.h"
#include <engine/Vectormath_Defines.hpp>
#include <engine/Hamiltonian.hpp>
#include <data/Geometry.hpp>

namespace Engine
{
	/*
		The Anisotropic Hamiltonian contains all information on the interactions between spins.
		The information is presented in index lists and parameter lists in order to easily calculate the energy of the system via summation.
	*/
	class Hamiltonian_Anisotropic : public Hamiltonian
	{
	public:
		// Constructor
		Hamiltonian_Anisotropic(
			scalarfield mu_s,
			intfield external_field_index, scalarfield external_field_magnitude, vectorfield external_field_normal,
			intfield anisotropy_index, scalarfield anisotropy_magnitude, vectorfield anisotropy_normal,
			std::vector<indexPairs> Exchange_indices, std::vector<scalarfield> Exchange_magnitude,
			std::vector<indexPairs> DMI_indices, std::vector<scalarfield> DMI_magnitude, std::vector<vectorfield> DMI_normal,
			std::vector<indexPairs> DD_indices, std::vector<scalarfield> DD_magnitude, std::vector<vectorfield> DD_normal,
			std::vector<indexQuadruplets> quadruplet_indices, std::vector<scalarfield> quadruplet_magnitude,
			std::vector<bool> boundary_conditions
		);

		void Update_Energy_Contributions() override;

		void Hessian(const vectorfield & spins, MatrixX & hessian) override;
		void Effective_Field(const vectorfield & spins, vectorfield & field) override;
		void Hamiltonian_Anisotropic::Energy_Contributions_per_Spin(const vectorfield & spins, std::vector<std::pair<std::string, scalarfield>> & contributions) override;

		// Hamiltonian name as string
		const std::string& Name() override;

		// ------------ General Variables ------------
		//std::vector<scalar> & field;
		
		// ------------ Single Spin Interactions ------------
		// Spin moment
		scalarfield mu_s;									// [nos]
		// External Magnetic Field
		intfield external_field_index;
		scalarfield external_field_magnitude;	// [nos]
		vectorfield external_field_normal;		// [nos] (x, y, z)
		// Anisotropy
		intfield anisotropy_index;
		scalarfield anisotropy_magnitude;		// [nos]
		vectorfield anisotropy_normal;			// [nos] (x, y, z)

		// ------------ Pair Interactions ------------
		// Exchange interaction
		std::vector<indexPairs> Exchange_indices;		// [periodicity][nop][2] (i,j)
		std::vector<scalarfield> Exchange_magnitude;	// [periodicity][nop]    J_ij
																// DMI
		std::vector<indexPairs> DMI_indices;			// [periodicity][nop][2] (i,j)
		std::vector<scalarfield> DMI_magnitude;			// [periodicity][nop]    D_ij
		std::vector<vectorfield> DMI_normal;			// [periodicity][nop][3] (Dx,Dy,Dz)
		// Dipole Dipole interaction
		std::vector<indexPairs> DD_indices;			// [periodicity][nop][2] (i,j)
		std::vector<scalarfield> DD_magnitude;			// [periodicity][nop]    r_ij (distance)
		std::vector<vectorfield> DD_normal;			// [periodicity][nop][4] (nx,ny,nz)

		// ------------ Quadruplet Interactions ------------
		std::vector<indexQuadruplets> Quadruplet_indices;
		std::vector<scalarfield> Quadruplet_magnitude;

	private:
		// ------------ Effective Field Functions ------------
		// Calculate the Zeeman effective field of a single Spin
		void Field_Zeeman(const vectorfield & spins, vectorfield & eff_field);
		// Calculate the Anisotropy effective field of a single Spin
		void Field_Anisotropy(const vectorfield & spins, vectorfield & eff_field);
		// Calculate the exchange interaction effective field of a Spin Pair
		void Field_Exchange(const vectorfield & spins, indexPairs & indices, scalarfield & J_ij, vectorfield & eff_field);
		// Calculate the DMI effective field of a Spin Pair
		void Field_DMI(const vectorfield & spins, indexPairs & indices, scalarfield & DMI_magnitude, vectorfield & DMI_normal, vectorfield & eff_field);
		// Calculates the Dipole-Dipole contribution to the effective field of spin ispin within system s
		void Field_DD(const vectorfield& spins, indexPairs & indices, scalarfield & DD_magnitude, vectorfield & DD_normal, vectorfield & eff_field);
		// Quadruplet
		void Field_Quadruplet(const vectorfield & spins, indexQuadruplets & indices, scalarfield & magnitude, vectorfield & eff_field);

		// ------------ Energy Functions ------------
		// Indices for Energy vector
		int idx_zeeman, idx_anisotropy, idx_exchange, idx_dmi, idx_dd, idx_quadruplet;
		// Calculate the Zeeman energy of a Spin System
		void E_Zeeman(const vectorfield & spins, scalarfield & Energy);
		// Calculate the Anisotropy energy of a Spin System
		void E_Anisotropy(const vectorfield & spins, scalarfield & Energy);
		// Calculate the exchange interaction energy of a Spin System
		void E_Exchange(const vectorfield & spins, indexPairs & indices, scalarfield & J_ij, scalarfield & Energy);
		// Calculate the DMI energy of a Spin System
		void E_DMI(const vectorfield & spins, indexPairs & indices, scalarfield & DMI_magnitude, vectorfield & DMI_normal, scalarfield & Energy);
		// calculates the Dipole-Dipole Energy
		void E_DD(const vectorfield& spins, indexPairs & indices, scalarfield & DD_magnitude, vectorfield & DD_normal, scalarfield & Energy);
		// Quadruplet
		void E_Quadruplet(const vectorfield & spins, indexQuadruplets & indices, scalarfield & magnitude, scalarfield & Energy);

	};
}
#endif