#pragma once
#ifndef HAMILTONIAN_ANISOTROPIC_NEW_H
#define HAMILTONIAN_ANISOTROPIC_NEW_H

#include <vector>

#include "Core_Defines.h"
#include "Hamiltonian.hpp"
#include "Geometry.hpp"

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
			std::vector<scalar> mu_s,
			std::vector<scalar> external_field_magnitude, std::vector<std::vector<scalar>> external_field_normal,
			std::vector<int> anisotropy_index, std::vector<scalar> anisotropy_magnitude, std::vector<std::vector<scalar>> anisotropy_normal,
			std::vector<std::vector<std::vector<int>>> Exchange_indices, std::vector<std::vector<scalar>> Exchange_magnitude,
			std::vector<std::vector<std::vector<int>>> DMI_indices, std::vector<std::vector<scalar>> DMI_magnitude, std::vector<std::vector<std::vector<scalar>>> DMI_normal,
			std::vector<std::vector<std::vector<int>>> BQC_indices, std::vector<std::vector<scalar>> BQC_magnitude,
			std::vector<std::vector<std::vector<int>>> DD_indices, std::vector<std::vector<scalar>> DD_magnitude, std::vector<std::vector<std::vector<scalar>>> DD_normal,
			std::vector<bool> boundary_conditions
		);

		void Hessian(const std::vector<scalar> & spins, std::vector<scalar> & hessian) override;
		void Effective_Field(const std::vector<scalar> & spins, std::vector<scalar> & field) override;
		scalar Energy(const std::vector<scalar> & spins) override;
		std::vector<scalar> Energy_Array(const std::vector<scalar> & spins) override;
		//std::vector<std::vector<scalar>> Energy_Array_per_Spin(std::vector<scalar> & spins) override;

		// Hamiltonian name as string
		const std::string& Name() override;

		// ------------ General Variables ------------
		//std::vector<scalar> & field;
		
		// ------------ Single Spin Interactions ------------
		// Spin moment
		std::vector<scalar> mu_s;									// [nos]
		// External Magnetic Field
		std::vector<scalar> external_field_magnitude;				// [nos]
		std::vector<std::vector<scalar>> external_field_normal;		// [3][nos] (x, y, z)
		// Anisotropy
		std::vector<int> anisotropy_index;
		std::vector<scalar> anisotropy_magnitude;					// [nos]
		std::vector<std::vector<scalar>> anisotropy_normal;			// [3][nos] (x, y, z)

		// ------------ Two Spin Interactions ------------
		// Exchange interaction
		std::vector<std::vector<std::vector<int>>> Exchange_indices;	// [periodicity][nop][2] (i,j)
		std::vector<std::vector<scalar>> Exchange_magnitude;			// [periodicity][nop]    J_ij
																		// DMI
		std::vector<std::vector<std::vector<int>>> DMI_indices;			// [periodicity][nop][2] (i,j)
		std::vector<std::vector<scalar>> DMI_magnitude;					// [periodicity][nop]    D_ij
		std::vector<std::vector<std::vector<scalar>>> DMI_normal;		// [periodicity][nop][3] (Dx,Dy,Dz)
		// Biquadratic Coupling
		std::vector<std::vector<std::vector<int>>> BQC_indices;			// [periodicity][nop][2] (i,j)
		std::vector<std::vector<scalar>> BQC_magnitude;					// [periodicity][nop]    Bij
		// Dipole Dipole interaction
		std::vector<std::vector<std::vector<int>>> DD_indices;			// [periodicity][nop][2] (i,j)
		std::vector<std::vector<scalar>> DD_magnitude;					// [periodicity][nop]    r_ij (distance)
		std::vector<std::vector<std::vector<scalar>>> DD_normal;		// [periodicity][nop][4] (nx,ny,nz)

		// ------------ Three Spin Interactions ------------
		//scalar TSI;
		//vector<Spin_Triplet> triplets;

		// ------------ Four Spin Interactions ------------
		//scalar kijkl;
		//vector<Spin_Quadruplets> quadruplets;
	private:
		// ------------ Effective Field Functions ------------
		// Calculate the Zeeman effective field of a single Spin
		void Field_Zeeman(int nos, const std::vector<scalar> & spins, std::vector<scalar> & eff_field, const int ispin);
		// Calculate the Anisotropy effective field of a single Spin
		void Field_Anisotropy(int nos, const std::vector<scalar> & spins, std::vector<scalar> & eff_field);
		// Calculate the exchange interaction effective field of a Spin Pair
		void Field_Exchange(int nos, const std::vector<scalar> & spins, std::vector<int> & indices, scalar J_ij, std::vector<scalar> & eff_field);
		// Calculate the DMI effective field of a Spin Pair
		void Field_DMI(int nos, const std::vector<scalar> & spins, std::vector<int> & indices, scalar & DMI_magnitude, std::vector<scalar> & DMI_normal, std::vector<scalar> & eff_field);
		// Calculate the BQC effective field of a Spin Pair
		void Field_BQC(int nos, const std::vector<scalar> & spins, std::vector<int> & indices, scalar B_ij, std::vector<scalar> & eff_field);
		// Calculates the Dipole-Dipole contribution to the effective field of spin ispin within system s
		void Field_DD(int nos, const std::vector<scalar>& spins, std::vector<int> & indices, scalar & DD_magnitude, std::vector<scalar> & DD_normal, std::vector<scalar> & eff_field);

		// ------------ Energy Functions ------------
		// Calculate the Zeeman energy of a Spin System
		void E_Zeeman(int nos, const std::vector<scalar> & spins, int ispin, std::vector<scalar> & Energy);
		// Calculate the Anisotropy energy of a Spin System
		void E_Anisotropy(int nos, const std::vector<scalar> & spins, std::vector<scalar> & Energy);
		// Calculate the exchange interaction energy of a Spin System
		void E_Exchange(int nos, const std::vector<scalar> & spins, std::vector<int> & indices, scalar J_ij, std::vector<scalar> & Energy);
		// Calculate the DMI energy of a Spin System
		void E_DMI(int nos, const std::vector<scalar> & spins, std::vector<int> & indices, scalar & DMI_magnitude, std::vector<scalar> & DMI_normal, std::vector<scalar> & Energy);
		// Calculate the BQC energy of a Spin System
		void E_BQC(int nos, const std::vector<scalar> & spins, std::vector<int> & indices, scalar B_ij, std::vector<scalar> & Energy);
		// calculates the Dipole-Dipole Energy of spin ispin within system s
		void E_DD(int nos, const std::vector<scalar>& spins, std::vector<int> & indices, scalar & DD_magnitude, std::vector<scalar> & DD_normal, std::vector<scalar> & Energy);

	};
}
#endif