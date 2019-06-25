#pragma once
#ifndef HAMILTONIAN_MICROMAGNETIC_H
#define HAMILTONIAN_MICROMAGNETIC_H

#include <vector>
#include <memory>

#include "Spirit_Defines.h"
#include <engine/Vectormath_Defines.hpp>
#include <engine/Hamiltonian.hpp>
#include <data/Geometry.hpp>

namespace Engine
{
    // enum class DDI_Method
    // {
    //     FFT    = SPIRIT_DDI_METHOD_FFT,
    //     FMM    = SPIRIT_DDI_METHOD_FMM,
    //     Cutoff = SPIRIT_DDI_METHOD_CUTOFF,
    //     None   = SPIRIT_DDI_METHOD_NONE
    // };

    /*
        The Micromagnetic Hamiltonian
    */
    class Hamiltonian_Micromagnetic : public Hamiltonian
    {
    public:
        Hamiltonian_Micromagnetic(
            scalar external_field_magnitude, Vector3 external_field_normal,
            Matrix3 anisotropy_tensor,
            scalar exchange_constant,
            scalar dmi_constant,
            std::shared_ptr<Data::Geometry> geometry,
            int spatial_gradient_order,
            intfield boundary_conditions
        );

        Hamiltonian_Micromagnetic(
            scalar external_field_magnitude, Vector3 external_field_normal,
            Matrix3 anisotropy_tensor,
            Matrix3 exchange_tensor,
            Matrix3 dmi_tensor,
            std::shared_ptr<Data::Geometry> geometry,
            int spatial_gradient_order,
            intfield boundary_conditions
        );

        void Update_Interactions();

        void Update_Energy_Contributions() override;

        void Hessian(const vectorfield & spins, MatrixX & hessian) override;
        void Gradient(const vectorfield & spins, vectorfield & gradient) override;
        void Energy_Contributions_per_Spin(const vectorfield & spins, std::vector<std::pair<std::string, scalarfield>> & contributions) override;
		void Energy_Update(const vectorfield & spins, std::vector<std::pair<std::string, scalarfield>> & contributions, vectorfield & gradient);
        // Calculate the total energy for a single spin to be used in Monte Carlo.
        //      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
        scalar Energy_Single_Spin(int ispin, const vectorfield & spins) override;

        // Hamiltonian name as string
        const std::string& Name() override;

        std::shared_ptr<Data::Geometry> geometry;

        // ------------ ... ------------
        int spatial_gradient_order;

        // ------------ Single Spin Interactions ------------
        // External magnetic field across the sample
        scalar external_field_magnitude;
        Vector3 external_field_normal;
        Matrix3 anisotropy_tensor;

        // ------------ Pair Interactions ------------
        // Exchange interaction
        Matrix3 exchange_tensor;
        // DMI
        Matrix3 dmi_tensor;
		neighbourfield neigh;
		field<Matrix3> spatial_gradient;
		bool A_is_nondiagonal=true;

    private:
        // ------------ Effective Field Functions ------------
        // Calculate the Zeeman effective field of a single Spin
        void Gradient_Zeeman(vectorfield & gradient);
        // Calculate the Anisotropy effective field of a single Spin
        void Gradient_Anisotropy(const vectorfield & spins, vectorfield & gradient);
        // Calculate the exchange interaction effective field of a Spin Pair
        void Gradient_Exchange(const vectorfield & spins, vectorfield & gradient);
        // Calculate the DMI effective field of a Spin Pair
        void Gradient_DMI(const vectorfield & spins, vectorfield & gradient);
		void Spatial_Gradient(const vectorfield & spins);

        // ------------ Energy Functions ------------
        // Indices for Energy vector
        int idx_zeeman, idx_anisotropy, idx_exchange, idx_dmi, idx_ddi;
		void E_Update(const vectorfield & spins, scalarfield & Energy, vectorfield & gradient);
        // Calculate the Zeeman energy of a Spin System
        void E_Zeeman(const vectorfield & spins, scalarfield & Energy);
        // Calculate the Anisotropy energy of a Spin System
        void E_Anisotropy(const vectorfield & spins, scalarfield & Energy);
        // Calculate the exchange interaction energy of a Spin System
        void E_Exchange(const vectorfield & spins, scalarfield & Energy);
        // Calculate the DMI energy of a Spin System
        void E_DMI(const vectorfield & spins, scalarfield & Energy);
        // Dipolar interactions
        void E_DDI(const vectorfield & spins, scalarfield & Energy);
    };


}
#endif