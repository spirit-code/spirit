#pragma once
#ifndef SPIRIT_CORE_ENGINE_HAMILTONIAN_HEISENBERG_HPP
#define SPIRIT_CORE_ENGINE_HAMILTONIAN_HEISENBERG_HPP

#include <memory>
#include <vector>

#include "FFT.hpp"
#include "Spirit_Defines.h"
#include <Spirit/Hamiltonian.h>
#include <data/Geometry.hpp>
#include <engine/Hamiltonian.hpp>
#include <engine/Vectormath_Defines.hpp>

namespace Engine
{

enum class DDI_Method
{
    FFT    = SPIRIT_DDI_METHOD_FFT,
    FMM    = SPIRIT_DDI_METHOD_FMM,
    Cutoff = SPIRIT_DDI_METHOD_CUTOFF,
    None   = SPIRIT_DDI_METHOD_NONE
};

/*
    The Heisenberg Hamiltonian using Pairs contains all information on the interactions between spins.
    The information is presented in pair lists and parameter lists in order to easily e.g. calculate the energy of the
   system via summation. Calculations are made on a per-pair basis running over all pairs.
*/
class Hamiltonian_Heisenberg : public Hamiltonian
{
public:
    Hamiltonian_Heisenberg(
        scalar external_field_magnitude, Vector3 external_field_normal, intfield anisotropy_indices,
        scalarfield anisotropy_magnitudes, vectorfield anisotropy_normals, pairfield exchange_pairs,
        scalarfield exchange_magnitudes, pairfield dmi_pairs, scalarfield dmi_magnitudes, vectorfield dmi_normals,
        DDI_Method ddi_method, intfield ddi_n_periodic_images, bool ddi_pb_zero_padding, scalar ddi_radius,
        quadrupletfield quadruplets, scalarfield quadruplet_magnitudes, std::shared_ptr<Data::Geometry> geometry,
        intfield boundary_conditions );

    Hamiltonian_Heisenberg(
        scalar external_field_magnitude, Vector3 external_field_normal, intfield anisotropy_indices,
        scalarfield anisotropy_magnitudes, vectorfield anisotropy_normals, scalarfield exchange_shell_magnitudes,
        scalarfield dmi_shell_magnitudes, int dm_chirality, DDI_Method ddi_method, intfield ddi_n_periodic_images,
        bool ddi_pb_zero_padding, scalar ddi_radius, quadrupletfield quadruplets, scalarfield quadruplet_magnitudes,
        std::shared_ptr<Data::Geometry> geometry, intfield boundary_conditions );

    void Update_Interactions();

    void Update_Energy_Contributions() override;

    void Hessian( const vectorfield & spins, MatrixX & hessian ) override;
    void Sparse_Hessian( const vectorfield & spins, SpMatrixX & hessian ) override;

    void Gradient( const vectorfield & spins, vectorfield & gradient ) override;
    void Gradient_and_Energy( const vectorfield & spins, vectorfield & gradient, scalar & energy ) override;

    void Energy_Contributions_per_Spin(
        const vectorfield & spins, std::vector<std::pair<std::string, scalarfield>> & contributions ) override;

    // Calculate the total energy for a single spin to be used in Monte Carlo.
    //      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
    scalar Energy_Single_Spin( int ispin, const vectorfield & spins ) override;

    // Hamiltonian name as string
    const std::string & Name() override;

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
    scalarfield exchange_shell_magnitudes;
    pairfield exchange_pairs_in;
    scalarfield exchange_magnitudes_in;
    pairfield exchange_pairs;
    scalarfield exchange_magnitudes;
    // DMI
    scalarfield dmi_shell_magnitudes;
    int dmi_shell_chirality;
    pairfield dmi_pairs_in;
    scalarfield dmi_magnitudes_in;
    vectorfield dmi_normals_in;
    pairfield dmi_pairs;
    scalarfield dmi_magnitudes;
    vectorfield dmi_normals;
    // Dipole Dipole interaction
    DDI_Method ddi_method;
    intfield ddi_n_periodic_images;
    bool ddi_pb_zero_padding;
    //      ddi cutoff variables
    scalar ddi_cutoff_radius;
    pairfield ddi_pairs;
    scalarfield ddi_magnitudes;
    vectorfield ddi_normals;

    // ------------ Quadruplet Interactions ------------
    quadrupletfield quadruplets;
    scalarfield quadruplet_magnitudes;

    std::shared_ptr<Data::Geometry> geometry;

    // ------------ Effective Field Functions ------------
    // Calculate the Zeeman effective field of a single Spin
    void Gradient_Zeeman( vectorfield & gradient );
    // Calculate the Anisotropy effective field of a single Spin
    void Gradient_Anisotropy( const vectorfield & spins, vectorfield & gradient );
    // Calculate the exchange interaction effective field of a Spin Pair
    void Gradient_Exchange( const vectorfield & spins, vectorfield & gradient );
    // Calculate the DMI effective field of a Spin Pair
    void Gradient_DMI( const vectorfield & spins, vectorfield & gradient );
    // Calculates the Dipole-Dipole contribution to the effective field of spin ispin within system s
    void Gradient_DDI( const vectorfield & spins, vectorfield & gradient );

    // Quadruplet
    void Gradient_Quadruplet( const vectorfield & spins, vectorfield & gradient );

    // ------------ Energy Functions ------------
    // Getters for Indices of the energy vector
    inline int Idx_Zeeman()
    {
        return idx_zeeman;
    };
    inline int Idx_Anisotropy()
    {
        return idx_anisotropy;
    };
    inline int Idx_Exchange()
    {
        return idx_exchange;
    };
    inline int Idx_DMI()
    {
        return idx_dmi;
    };
    inline int Idx_DDI()
    {
        return idx_ddi;
    };
    inline int Idx_Quadruplet()
    {
        return idx_quadruplet;
    };

    // Calculate the Zeeman energy of a Spin System
    void E_Zeeman( const vectorfield & spins, scalarfield & Energy );
    // Calculate the Anisotropy energy of a Spin System
    void E_Anisotropy( const vectorfield & spins, scalarfield & Energy );
    // Calculate the exchange interaction energy of a Spin System
    void E_Exchange( const vectorfield & spins, scalarfield & Energy );
    // Calculate the DMI energy of a Spin System
    void E_DMI( const vectorfield & spins, scalarfield & Energy );
    // Calculate the Dipole-Dipole energy
    void E_DDI( const vectorfield & spins, scalarfield & Energy );
    // Calculate the Quadruplet energy
    void E_Quadruplet( const vectorfield & spins, scalarfield & Energy );

private:
    int idx_zeeman, idx_anisotropy, idx_exchange, idx_dmi, idx_ddi, idx_quadruplet;
    void Gradient_DDI_Cutoff( const vectorfield & spins, vectorfield & gradient );
    void Gradient_DDI_Direct( const vectorfield & spins, vectorfield & gradient );
    void Gradient_DDI_FFT( const vectorfield & spins, vectorfield & gradient );
    void E_DDI_Direct( const vectorfield & spins, scalarfield & Energy );
    void E_DDI_Cutoff( const vectorfield & spins, scalarfield & Energy );
    void E_DDI_FFT( const vectorfield & spins, scalarfield & Energy );

    // Preparations for DDI-Convolution Algorithm
    void Prepare_DDI();
    void Clean_DDI();

    // Plans for FT / rFT
    FFT::FFT_Plan fft_plan_spins;
    FFT::FFT_Plan fft_plan_reverse;

    field<FFT::FFT_cpx_type> transformed_dipole_matrices;

    bool save_dipole_matrices = false;
    field<FFT::FFT_real_type> dipole_matrices;

    // Number of inter-sublattice contributions
    int n_inter_sublattice;
    // At which index to look up the inter-sublattice D-matrices
    field<int> inter_sublattice_lookup;

    // Lengths of padded system
    field<int> n_cells_padded;
    // Total number of padded spins per sublattice
    int sublattice_size;

    FFT::StrideContainer spin_stride;
    FFT::StrideContainer dipole_stride;

    // Calculate the FT of the padded D-matrics
    void FFT_Dipole_Matrices( FFT::FFT_Plan & fft_plan_dipole, int img_a, int img_b, int img_c );
    // Calculate the FT of the padded spins
    void FFT_Spins( const vectorfield & spins );

    // Bounds for nested for loops. Only important for the CUDA version
    field<int> it_bounds_pointwise_mult;
    field<int> it_bounds_write_gradients;
    field<int> it_bounds_write_spins;
    field<int> it_bounds_write_dipole;
};

} // namespace Engine

#endif