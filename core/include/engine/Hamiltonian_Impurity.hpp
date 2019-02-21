#pragma once
#ifndef HAMILTONIAN_IMPURITY_H
#define HAMILTONIAN_IMPURITY_H

#include <vector>
#include <memory>

#include "Spirit_Defines.h"
#include <engine/Vectormath_Defines.hpp>
#include <engine/Hamiltonian.hpp>
#include <data/Geometry.hpp>
#include <Spirit/Hamiltonian.h>

namespace Engine
{
    /*
        The Heisenberg Hamiltonian for an impurity cluster.
        This is meant to only be used inside another Heisenberg Hamiltonian to modulate
        the interactions in a given region and handle additional atoms (e.g. adatoms).
    */
    class Hamiltonian_Impurity : public Hamiltonian
    {
    public:
        Hamiltonian_Impurity(
            scalar external_field_magnitude, Vector3 external_field_normal,
            intfield anisotropy_indices, scalarfield anisotropy_magnitudes, vectorfield anisotropy_normals,
            pairfield exchange_pairs, scalarfield exchange_magnitudes,
            pairfield dmi_pairs, scalarfield dmi_magnitudes, vectorfield dmi_normals,
            std::shared_ptr<Data::Geometry> geometry,
            intfield boundary_conditions
        );

        // void Update_Interactions();

        void Update_Energy_Contributions() override;

        void Hessian(const vectorfield & spins, MatrixX & hessian) override;
    //     void Gradient(const vectorfield & spins, vectorfield & gradient) override;
    //     void Energy_Contributions_per_Spin(const vectorfield & spins, std::vector<std::pair<std::string, scalarfield>> & contributions) override;

    //     // Calculate the total energy for a single spin to be used in Monte Carlo.
    //     //      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
        scalar Energy_Single_Spin(int ispin, const vectorfield & spins) override;

    //     // Hamiltonian name as string
        const std::string& Name() override;
        
        // ------------ Single Spin Interactions ------------
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
        pairfield   exchange_pairs;
        scalarfield exchange_magnitudes;
        // DMI
        pairfield   dmi_pairs;
        scalarfield dmi_magnitudes;
        vectorfield dmi_normals;

        // ------------ Effective Field Functions ------------
        // Calculate the Zeeman effective field of a single Spin
        void Gradient_Zeeman(vectorfield & gradient);
        // Calculate the Anisotropy effective field of a single Spin
        void Gradient_Anisotropy(const vectorfield & spins, vectorfield & gradient);
        // Calculate the exchange interaction effective field of a Spin Pair
        void Gradient_Exchange(const vectorfield & spins, vectorfield & gradient);
        // Calculate the DMI effective field of a Spin Pair
        void Gradient_DMI(const vectorfield & spins, vectorfield & gradient);

        // ------------ Energy Functions ------------
        // Calculate the Zeeman energy of a Spin System
        void E_Zeeman(const vectorfield & spins, scalarfield & Energy);
        // Calculate the Anisotropy energy of a Spin System
        void E_Anisotropy(const vectorfield & spins, scalarfield & Energy);
        // Calculate the exchange interaction energy of a Spin System
        void E_Exchange(const vectorfield & spins, scalarfield & Energy);
        // Calculate the DMI energy of a Spin System
        void E_DMI(const vectorfield & spins, scalarfield & Energy);

    private:
        std::shared_ptr<Data::Geometry> geometry;

    //     // Indices for Energy vector
    //     int idx_zeeman, idx_anisotropy, idx_exchange, idx_dmi, idx_ddi;
    };


}
#endif