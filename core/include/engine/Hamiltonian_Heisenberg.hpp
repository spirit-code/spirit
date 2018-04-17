#pragma once
#ifndef HAMILTONIAN_HEISENBERG_H
#define HAMILTONIAN_HEISENBERG_H

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
	class Hamiltonian_Heisenberg : public Hamiltonian
	{
	public:
        Hamiltonian_Heisenberg(
            scalarfield mu_s,
            scalar external_field_magnitude, Vector3 external_field_normal,
            intfield anisotropy_indices, scalarfield anisotropy_magnitudes, vectorfield anisotropy_normals,
            pairfield exchange_pairs, scalarfield exchange_magnitudes,
            pairfield dmi_pairs, scalarfield dmi_magnitudes, vectorfield dmi_normals,
            scalar ddi_radius,
            quadrupletfield quadruplets, scalarfield quadruplet_magnitudes,
            std::shared_ptr<Data::Geometry> geometry,
            intfield boundary_conditions
        );

		Hamiltonian_Heisenberg(
			scalarfield mu_s,
			scalar external_field_magnitude, Vector3 external_field_normal,
			intfield anisotropy_indices, scalarfield anisotropy_magnitudes, vectorfield anisotropy_normals,
			scalarfield exchange_magnitudes,
			scalarfield dmi_magnitudes, int dm_chirality,
			scalar ddi_radius,
        	quadrupletfield quadruplets, scalarfield quadruplet_magnitudes,
			std::shared_ptr<Data::Geometry> geometry,
			intfield boundary_conditions
		);

		void Update_Energy_Contributions() override;

		void Hessian(const vectorfield & spins, MatrixX & hessian) override;
		void Gradient(const vectorfield & spins, vectorfield & gradient) override;
		void Energy_Contributions_per_Spin(const vectorfield & spins, std::vector<std::pair<std::string, scalarfield>> & contributions) override;

		// Hamiltonian name as string
		const std::string& Name() override;
		
		// ------------ Single Spin Interactions ------------
		// Spin moments of basis cell atoms
		scalarfield mu_s;
        // External magnetic field across the sample
        scalar external_field_magnitude;
        Vector3 external_field_normal;
		// External magnetic field - for now external magnetic field is homogeneous
        // If required, an additional, inhomogeneous external field should be added
		//   scalarfield external_field_magnitudes;
		//   vectorfield external_field_normals;
		// Anisotropy axes of a basis cell
        // (indexed, as any atom of the basis cell can have one or more anisotropy axes)
		intfield anisotropy_indices;
		scalarfield anisotropy_magnitudes;
		vectorfield anisotropy_normals;

		// ------------ Pair Interactions ------------
		// Exchange interaction
		int         exchange_n_shells;
		pairfield   exchange_pairs;
		scalarfield exchange_magnitudes;
		// DMI
		int         dmi_n_shells;
		pairfield   dmi_pairs;
		scalarfield dmi_magnitudes;
        vectorfield dmi_normals;
		// Dipole Dipole interaction
		scalar      ddi_radius;
		pairfield   ddi_pairs;
		scalarfield ddi_magnitudes;
		vectorfield ddi_normals;

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
		int idx_zeeman, idx_anisotropy, idx_exchange, idx_dmi, idx_ddi, idx_quadruplet;
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