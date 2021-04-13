#ifndef SPIRIT_USE_CUDA

#include <data/Spin_System.hpp>
#include <engine/Demagnetization_Tensor.hpp>
#include <engine/Hamiltonian_Micromagnetic.hpp>
#include <engine/Neighbours.hpp>
#include <engine/Vectormath.hpp>

#include <utility/Constants.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <algorithm>
#include <iostream>

using namespace Data;
using namespace Utility;
namespace C = Utility::Constants;
using Engine::Vectormath::check_atom_type;
using Engine::Vectormath::idx_from_pair;
using Engine::Vectormath::idx_from_tupel;

namespace Engine
{

Hamiltonian_Micromagnetic::Hamiltonian_Micromagnetic(
    scalar Ms, scalar external_field_magnitude, Vector3 external_field_normal, Matrix3 anisotropy_tensor,
    Matrix3 exchange_tensor, Matrix3 dmi_tensor, DDI_Method ddi_method, intfield ddi_n_periodic_images,
    scalar ddi_radius, std::shared_ptr<Data::Geometry> geometry, int spatial_gradient_order,
    intfield boundary_conditions )
        : Hamiltonian( boundary_conditions ),
          Ms( Ms ),
          spatial_gradient_order( spatial_gradient_order ),
          geometry( geometry ),
          external_field_magnitude( external_field_magnitude ),
          external_field_normal( external_field_normal ),
          anisotropy_tensor( anisotropy_tensor ),
          exchange_tensor( exchange_tensor ),
          dmi_tensor( dmi_tensor ),
          ddi_method( ddi_method ),
          ddi_n_periodic_images( ddi_n_periodic_images ),
          ddi_cutoff_radius( ddi_radius ),
          fft_plan_reverse( FFT::FFT_Plan() ),
          fft_plan_spins( FFT::FFT_Plan() )
{
    // Generate interaction pairs, constants etc.
    this->Update_Interactions();
}

void Hamiltonian_Micromagnetic::Update_Interactions()
{
#if defined( SPIRIT_USE_OPENMP )
    // When parallelising (cuda or openmp), we need all neighbours per spin
    const bool use_redundant_neighbours = true;
#else
    // When running on a single thread, we can ignore redundant neighbours
    const bool use_redundant_neighbours = false;
#endif

    // TODO: make sure that the geometry can be treated with this model:
    //       - rectilinear, only one "atom" per cell
    // if( geometry->n_cell_atoms != 1 )
    //     Log(...)

    // TODO: generate neighbour information for pairwise interactions

    // TODO: prepare dipolar interactions

    // Update, which terms still contribute

    neigh = pairfield( 0 );
    Neighbour neigh_tmp;
    neigh_tmp.i         = 0;
    neigh_tmp.j         = 0;
    neigh_tmp.idx_shell = 0;
    // order x -x y -y z -z xy (-x)(-y) x(-y) (-x)y xz (-x)(-z) x(-z) (-x)z yz (-y)(-z) y(-z) (-y)z results in 9 parts of Hessian
    neigh_tmp.translations[0] = 1;
    neigh_tmp.translations[1] = 0;
    neigh_tmp.translations[2] = 0;
    neigh.push_back( neigh_tmp );

    neigh_tmp.translations[0] = -1;
    neigh_tmp.translations[1] = 0;
    neigh_tmp.translations[2] = 0;
    neigh.push_back( neigh_tmp );

    neigh_tmp.translations[0] = 0;
    neigh_tmp.translations[1] = 1;
    neigh_tmp.translations[2] = 0;
    neigh.push_back( neigh_tmp );

    neigh_tmp.translations[0] = 0;
    neigh_tmp.translations[1] = -1;
    neigh_tmp.translations[2] = 0;
    neigh.push_back( neigh_tmp );

    neigh_tmp.translations[0] = 0;
    neigh_tmp.translations[1] = 0;
    neigh_tmp.translations[2] = 1;
    neigh.push_back( neigh_tmp );

    neigh_tmp.translations[0] = 0;
    neigh_tmp.translations[1] = 0;
    neigh_tmp.translations[2] = -1;
    neigh.push_back( neigh_tmp );

    neigh_tmp.translations[0] = 1;
    neigh_tmp.translations[1] = 1;
    neigh_tmp.translations[2] = 0;
    neigh.push_back( neigh_tmp );

    neigh_tmp.translations[0] = -1;
    neigh_tmp.translations[1] = -1;
    neigh_tmp.translations[2] = 0;
    neigh.push_back( neigh_tmp );

    neigh_tmp.translations[0] = 1;
    neigh_tmp.translations[1] = -1;
    neigh_tmp.translations[2] = 0;
    neigh.push_back( neigh_tmp );

    neigh_tmp.translations[0] = -1;
    neigh_tmp.translations[1] = +1;
    neigh_tmp.translations[2] = 0;
    neigh.push_back( neigh_tmp );

    neigh_tmp.translations[0] = 1;
    neigh_tmp.translations[1] = 0;
    neigh_tmp.translations[2] = 1;
    neigh.push_back( neigh_tmp );

    neigh_tmp.translations[0] = -1;
    neigh_tmp.translations[1] = 0;
    neigh_tmp.translations[2] = -1;
    neigh.push_back( neigh_tmp );

    neigh_tmp.translations[0] = 1;
    neigh_tmp.translations[1] = 0;
    neigh_tmp.translations[2] = -1;
    neigh.push_back( neigh_tmp );

    neigh_tmp.translations[0] = -1;
    neigh_tmp.translations[1] = 0;
    neigh_tmp.translations[2] = 1;
    neigh.push_back( neigh_tmp );

    neigh_tmp.translations[0] = 0;
    neigh_tmp.translations[1] = 1;
    neigh_tmp.translations[2] = 1;
    neigh.push_back( neigh_tmp );

    neigh_tmp.translations[0] = 0;
    neigh_tmp.translations[1] = -1;
    neigh_tmp.translations[2] = -1;
    neigh.push_back( neigh_tmp );

    neigh_tmp.translations[0] = 0;
    neigh_tmp.translations[1] = 1;
    neigh_tmp.translations[2] = -1;
    neigh.push_back( neigh_tmp );

    neigh_tmp.translations[0] = 0;
    neigh_tmp.translations[1] = -1;
    neigh_tmp.translations[2] = 1;
    neigh.push_back( neigh_tmp );

    this->spatial_gradient = field<Matrix3>( geometry->nos, Matrix3::Zero() );
    this->Prepare_DDI();
    this->Update_Energy_Contributions();
}

void Hamiltonian_Micromagnetic::Update_Energy_Contributions()
{
    this->energy_contributions_per_spin = std::vector<std::pair<std::string, scalarfield>>( 0 );

    // External field
    if( this->external_field_magnitude > 0 )
    {
        this->energy_contributions_per_spin.push_back( { "Zeeman", scalarfield( 0 ) } );
        this->idx_zeeman = this->energy_contributions_per_spin.size() - 1;
    }
    else
        this->idx_zeeman = -1;

    if( anisotropy_tensor.norm() == 0.0 )
    {
        this->energy_contributions_per_spin.push_back( { "Anisotropy", scalarfield( 0 ) } );
        this->idx_anisotropy = this->energy_contributions_per_spin.size() - 1;
    }
    else
        this->idx_anisotropy = -1;

    if( exchange_tensor.norm() == 0.0 )
    {
        this->energy_contributions_per_spin.push_back( { "Exchange", scalarfield( 0 ) } );
        this->idx_exchange = this->energy_contributions_per_spin.size() - 1;
    }
    else
        this->idx_exchange = -1;

    if( dmi_tensor.norm() == 0.0 )
    {
        this->energy_contributions_per_spin.push_back( { "DMI", scalarfield( 0 ) } );
        this->idx_dmi = this->energy_contributions_per_spin.size() - 1;
    }
    else
        this->idx_dmi = -1;

    if( this->ddi_method != DDI_Method::None )
    {
        this->energy_contributions_per_spin.push_back( { "DDI", scalarfield( 0 ) } );
        this->idx_ddi = this->energy_contributions_per_spin.size() - 1;
    }
    else
        this->idx_ddi = -1;
}

void Hamiltonian_Micromagnetic::Energy_Contributions_per_Spin(
    const vectorfield & spins, std::vector<std::pair<std::string, scalarfield>> & contributions )
{
    if( contributions.size() != this->energy_contributions_per_spin.size() )
    {
        contributions = this->energy_contributions_per_spin;
    }

    int nos = spins.size();
    for( auto & contrib : contributions )
    {
        // Allocate if not already allocated
        if( contrib.second.size() != nos )
            contrib.second = scalarfield( nos, 0 );
        // Otherwise set to zero
        else
            Vectormath::fill( contrib.second, 0 );
    }

    // External field
    if( this->idx_zeeman >= 0 )
        E_Zeeman( spins, contributions[idx_zeeman].second );

    // Anisotropy
    if( this->idx_anisotropy >= 0 )
        E_Anisotropy( spins, contributions[idx_anisotropy].second );

    // Exchange
    if( this->idx_exchange >= 0 )
        E_Exchange( spins, contributions[idx_exchange].second );

    // DMI
    if( this->idx_dmi >= 0 )
        E_DMI( spins, contributions[idx_dmi].second );

    // DDI
    if( this->idx_ddi >= 0 )
        E_DDI( spins, contributions[idx_ddi].second );
}

void Hamiltonian_Micromagnetic::E_Zeeman( const vectorfield & spins, scalarfield & Energy )
{
    auto & mu_s = this->geometry->mu_s;

#pragma omp parallel for
    for( int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        if( check_atom_type( this->geometry->atom_types[icell] ) )
            Energy[icell]
                -= mu_s[icell] * this->external_field_magnitude * this->external_field_normal.dot( spins[icell] );
    }
}

void Hamiltonian_Micromagnetic::E_Update( const vectorfield & spins, scalarfield & Energy, vectorfield & gradient )
{
#pragma omp parallel for
    for( int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        Energy[icell] -= 0.5 * Ms * gradient[icell].dot( spins[icell] );
    }
}

void Hamiltonian_Micromagnetic::E_Anisotropy( const vectorfield & spins, scalarfield & Energy ) {}

void Hamiltonian_Micromagnetic::E_Exchange( const vectorfield & spins, scalarfield & Energy ) {}

void Hamiltonian_Micromagnetic::E_DMI( const vectorfield & spins, scalarfield & Energy ) {}

void Hamiltonian_Micromagnetic::E_DDI( const vectorfield & spins, scalarfield & Energy )
{
    if( this->ddi_method == DDI_Method::FFT )
        this->E_DDI_FFT( spins, Energy );
}

scalar Hamiltonian_Micromagnetic::Energy_Single_Spin( int ispin, const vectorfield & spins )
{
    scalar Energy = 0;
    return Energy;
}

void Hamiltonian_Micromagnetic::Gradient( const vectorfield & spins, vectorfield & gradient )
{

    // Set to zero
    Vectormath::fill( gradient, { 0, 0, 0 } );
    this->Spatial_Gradient( spins );

    // External field
    this->Gradient_Zeeman( gradient );

    // Anisotropy
    this->Gradient_Anisotropy( spins, gradient );

    // Exchange
    this->Gradient_Exchange( spins, gradient );

    // DMI
    this->Gradient_DMI( spins, gradient );

    // DDI
    this->Gradient_DDI( spins, gradient );

    // double energy=0;
    // #pragma omp parallel for reduction(-:energy)
    // for( int icell = 0; icell < geometry->n_cells_total; ++icell )
    // {
    //     energy -= 0.5 * Ms * gradient[icell].dot(spins[icell]);
    // }
    // printf("Energy total: %f\n", energy/ geometry->n_cells_total);
}

void Hamiltonian_Micromagnetic::Gradient_Zeeman( vectorfield & gradient )
{
    // In this context this is magnetisation per cell
    auto & mu_s = this->geometry->mu_s;

#pragma omp parallel for
    for( int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        if( check_atom_type( this->geometry->atom_types[icell] ) )
            gradient[icell] -= mu_s[icell] * C::mu_B * this->external_field_magnitude * this->external_field_normal;
    }
}

void Hamiltonian_Micromagnetic::Gradient_Anisotropy( const vectorfield & spins, vectorfield & gradient )
{
    Vector3 temp1{ 1, 0, 0 };
    Vector3 temp2{ 0, 1, 0 };
    Vector3 temp3{ 0, 0, 1 };
#pragma omp parallel for
    for( int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        // for( int iani = 0; iani < 1; ++iani )
        // {
        // gradient[icell] -= 2.0 * 8.44e6 / Ms * temp3 * temp3.dot(spins[icell]);
        gradient[icell] -= 2.0 * C::mu_B * anisotropy_tensor * spins[icell] / Ms;
        // gradient[icell] -= 2.0 * this->anisotropy_magnitudes[iani] / Ms * ((pow(temp2.dot(spins[icell]),2)+
        // pow(temp3.dot(spins[icell]), 2))*(temp1.dot(spins[icell])*temp1)+ (pow(temp1.dot(spins[icell]), 2) +
        // pow(temp3.dot(spins[icell]), 2))*(temp2.dot(spins[icell])*temp2)+(pow(temp1.dot(spins[icell]),2)+
        // pow(temp2.dot(spins[icell]), 2))*(temp3.dot(spins[icell])*temp3)); gradient[icell] += 2.0 * 50000 / Ms *
        // ((pow(temp2.dot(spins[icell]), 2) + pow(temp3.dot(spins[icell]), 2))*(temp1.dot(spins[icell])*temp1) +
        // (pow(temp1.dot(spins[icell]), 2) + pow(temp3.dot(spins[icell]), 2))*(temp2.dot(spins[icell])*temp2));
        // }
    }
}

void Hamiltonian_Micromagnetic::Gradient_Exchange( const vectorfield & spins, vectorfield & gradient )
{
    auto & delta = geometry->cell_size;
// scalar delta[3] = { 3e-10, 3e-10, 3e-9 };
// scalar delta[3] = { 277e-12, 277e-12, 277e-12 };

// nongradient implementation
/*
#pragma omp parallel for
for (unsigned int icell = 0; icell < geometry->n_cells_total; ++icell)
{
    int ispin = icell;//basically id of a cell
    for (unsigned int i = 0; i < 3; ++i)
    {

        int ispin_plus = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms,
geometry->atom_types, neigh[i]); int ispin_minus = idx_from_pair(ispin, boundary_conditions, geometry->n_cells,
geometry->n_cell_atoms, geometry->atom_types, neigh[i + 1]); if (ispin_plus == -1) { ispin_plus = ispin;
        }
        if (ispin_minus == -1) {
            ispin_minus = ispin;
        }
        gradient[ispin][i] -= exchange_tensor(i, i)*(spins[ispin_plus][i] - 2 * spins[ispin][i] + spins[ispin_minus][i])
/ (delta[i]) / (delta[i]);

    }
    if (A_is_nondiagonal == true) {
        int ispin_plus_plus = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms,
geometry->atom_types, neigh[6]); int ispin_minus_minus = idx_from_pair(ispin, boundary_conditions, geometry->n_cells,
geometry->n_cell_atoms, geometry->atom_types, neigh[7]); int ispin_plus_minus = idx_from_pair(ispin,
boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, neigh[8]); int ispin_minus_plus =
idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, neigh[9]);

        if (ispin_plus_plus == -1) {
            ispin_plus_plus = ispin;
        }
        if (ispin_minus_minus == -1) {
            ispin_minus_minus = ispin;
        }
        if (ispin_plus_minus == -1) {
            ispin_plus_minus = ispin;
        }
        if (ispin_minus_plus == -1) {
            ispin_minus_plus = ispin;
        }
        gradient[ispin][0] -= exchange_tensor(0, 1)*(spins[ispin_plus_plus][1] - spins[ispin_plus_minus][1] -
spins[ispin_minus_plus][1] + spins[ispin_minus_minus][1]) / (delta[0]) / (delta[1]) / 4; gradient[ispin][1] -=
exchange_tensor(1, 0)*(spins[ispin_plus_plus][0] - spins[ispin_plus_minus][0] - spins[ispin_minus_plus][0] +
spins[ispin_minus_minus][0]) / (delta[0]) / (delta[1]) / 4;

        ispin_plus_plus = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms,
geometry->atom_types, neigh[10]); ispin_minus_minus = idx_from_pair(ispin, boundary_conditions, geometry->n_cells,
geometry->n_cell_atoms, geometry->atom_types, neigh[11]); ispin_plus_minus = idx_from_pair(ispin, boundary_conditions,
geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, neigh[12]); ispin_minus_plus = idx_from_pair(ispin,
boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, neigh[13]);

        if (ispin_plus_plus == -1) {
            ispin_plus_plus = ispin;
        }
        if (ispin_minus_minus == -1) {
            ispin_minus_minus = ispin;
        }
        if (ispin_plus_minus == -1) {
            ispin_plus_minus = ispin;
        }
        if (ispin_minus_plus == -1) {
            ispin_minus_plus = ispin;
        }
        gradient[ispin][0] -= exchange_tensor(0, 2)*(spins[ispin_plus_plus][2] - spins[ispin_plus_minus][2] -
spins[ispin_minus_plus][2] + spins[ispin_minus_minus][2]) / (delta[0]) / (delta[2]) / 4; gradient[ispin][2] -=
exchange_tensor(2, 0)*(spins[ispin_plus_plus][0] - spins[ispin_plus_minus][0] - spins[ispin_minus_plus][0] +
spins[ispin_minus_minus][0]) / (delta[0]) / (delta[2]) / 4;

        ispin_plus_plus = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms,
geometry->atom_types, neigh[14]); ispin_minus_minus = idx_from_pair(ispin, boundary_conditions, geometry->n_cells,
geometry->n_cell_atoms, geometry->atom_types, neigh[15]); ispin_plus_minus = idx_from_pair(ispin, boundary_conditions,
geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, neigh[16]); ispin_minus_plus = idx_from_pair(ispin,
boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, neigh[17]);

        if (ispin_plus_plus == -1) {
            ispin_plus_plus = ispin;
        }
        if (ispin_minus_minus == -1) {
            ispin_minus_minus = ispin;
        }
        if (ispin_plus_minus == -1) {
            ispin_plus_minus = ispin;
        }
        if (ispin_minus_plus == -1) {
            ispin_minus_plus = ispin;
        }
        gradient[ispin][1] -= exchange_tensor(1, 2)*(spins[ispin_plus_plus][2] - spins[ispin_plus_minus][2] -
spins[ispin_minus_plus][2] + spins[ispin_minus_minus][2]) / (delta[1]) / (delta[2]) / 4; gradient[ispin][2] -=
exchange_tensor(2, 1)*(spins[ispin_plus_plus][1] - spins[ispin_plus_minus][1] - spins[ispin_minus_plus][1] +
spins[ispin_minus_minus][1]) / (delta[1]) / (delta[2]) / 4;
    }

}*/

// Gradient implementation
#pragma omp parallel for
    for( unsigned int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        for( unsigned int i = 0; i < 3; ++i )
        {
            int icell_plus = idx_from_pair(
                icell, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                neigh[2 * i] );
            int icell_minus = idx_from_pair(
                icell, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                neigh[2 * i + 1] );

            if( icell_plus >= 0 || icell_minus >= 0 )
            {
                if( icell_plus == -1 )
                    icell_plus = icell;
                if( icell_minus == -1 )
                    icell_minus = icell;

                gradient[icell] -= 2 * C::mu_B * exchange_tensor
                                   * ( spins[icell_plus] - 2 * spins[icell] + spins[icell_minus] )
                                   / ( Ms * delta[i] * delta[i] );
            }

            // gradient[icell][0] -= 2*exchange_tensor(i, i) / Ms * (spins[icell_plus][0] - 2*spins[icell][0] +
            // spins[icell_minus][0]) / (delta[i]) / (delta[i]); gradient[icell][1] -= 2*exchange_tensor(i, i) / Ms *
            // (spins[icell_plus][1] - 2*spins[icell][1] + spins[icell_minus][1]) / (delta[i]) / (delta[i]);
            // gradient[icell][2]
            // -= 2*exchange_tensor(i, i) / Ms * (spins[icell_plus][2] - 2*spins[icell][2] + spins[icell_minus][2]) /
            // (delta[i]) / (delta[i]);
        }
        // if( this->A_is_nondiagonal )
        // {
        //     int ispin = icell;

        //     // xy
        //     int ispin_right  = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms,
        //     geometry->atom_types, neigh[0]); int ispin_left   = idx_from_pair(ispin, boundary_conditions,
        //     geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, neigh[1]); int ispin_top    =
        //     idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms,
        //     geometry->atom_types, neigh[2]); int ispin_bottom = idx_from_pair(ispin, boundary_conditions,
        //     geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, neigh[3]);

        //     if( ispin_right == -1 )
        //         ispin_right = ispin;
        //     if( ispin_left == -1 )
        //         ispin_left = ispin;
        //     if( ispin_top == -1 )
        //         ispin_top = ispin;
        //     if( ispin_bottom == -1 )
        //         ispin_bottom = ispin;

        //     gradient[ispin][0] -= 2*exchange_tensor(0, 1) / Ms * ((spatial_gradient[ispin_top](0, 0) -
        //     spatial_gradient[ispin_bottom](0, 0)) / 4 / delta[1] + (spatial_gradient[ispin_right](0, 1) -
        //     spatial_gradient[ispin_left](0, 1)) / 4 / delta[0]); gradient[ispin][0] -= 2*exchange_tensor(1, 0) / Ms *
        //     ((spatial_gradient[ispin_top](0, 0) - spatial_gradient[ispin_bottom](0, 0)) / 4 / delta[1] +
        //     (spatial_gradient[ispin_right](0, 1) - spatial_gradient[ispin_left](0, 1)) / 4 / delta[0]);
        //     gradient[ispin][1]
        //     -= 2*exchange_tensor(0, 1) / Ms * ((spatial_gradient[ispin_top](1, 0) - spatial_gradient[ispin_bottom](1,
        //     0)) / 4 / delta[1] + (spatial_gradient[ispin_right](1, 1) - spatial_gradient[ispin_left](1, 1)) / 4 /
        //     delta[0]); gradient[ispin][1] -= 2*exchange_tensor(1, 0) / Ms * ((spatial_gradient[ispin_top](1, 0) -
        //     spatial_gradient[ispin_bottom](1, 0)) / 4 / delta[1] + (spatial_gradient[ispin_right](1, 1) -
        //     spatial_gradient[ispin_left](1, 1)) / 4 / delta[0]); gradient[ispin][2] -= 2*exchange_tensor(0, 1) / Ms *
        //     ((spatial_gradient[ispin_top](2, 0) - spatial_gradient[ispin_bottom](2, 0)) / 4 / delta[1] +
        //     (spatial_gradient[ispin_right](2, 1) - spatial_gradient[ispin_left](2, 1)) / 4 / delta[0]);
        //     gradient[ispin][2]
        //     -= 2*exchange_tensor(1, 0) / Ms * ((spatial_gradient[ispin_top](2, 0) - spatial_gradient[ispin_bottom](2,
        //     0)) / 4 / delta[1] + (spatial_gradient[ispin_right](2, 1) - spatial_gradient[ispin_left](2, 1)) / 4 /
        //     delta[0]);

        //     // xz
        //     ispin_right  = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms,
        //     geometry->atom_types, neigh[0]); ispin_left   = idx_from_pair(ispin, boundary_conditions,
        //     geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, neigh[1]); ispin_top    =
        //     idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms,
        //     geometry->atom_types, neigh[4]); ispin_bottom = idx_from_pair(ispin, boundary_conditions,
        //     geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, neigh[5]);

        //     if( ispin_right == -1 )
        //         ispin_right = ispin;
        //     if( ispin_left == -1 )
        //         ispin_left = ispin;
        //     if( ispin_top == -1 )
        //         ispin_top = ispin;
        //     if( ispin_bottom == -1 )
        //         ispin_bottom = ispin;

        //     gradient[ispin][0] -= 2*exchange_tensor(0, 2) / Ms * ((spatial_gradient[ispin_top](0, 0) -
        //     spatial_gradient[ispin_bottom](0, 0)) / 4 / delta[2] + (spatial_gradient[ispin_right](0, 2) -
        //     spatial_gradient[ispin_left](0, 2)) / 4 / delta[0]); gradient[ispin][0] -= 2*exchange_tensor(2, 0) / Ms *
        //     ((spatial_gradient[ispin_top](0, 0) - spatial_gradient[ispin_bottom](0, 0)) / 4 / delta[2] +
        //     (spatial_gradient[ispin_right](0, 2) - spatial_gradient[ispin_left](0, 2)) / 4 / delta[0]);
        //     gradient[ispin][1]
        //     -= 2*exchange_tensor(0, 2) / Ms * ((spatial_gradient[ispin_top](1, 0) - spatial_gradient[ispin_bottom](1,
        //     0)) / 4 / delta[2] + (spatial_gradient[ispin_right](1, 2) - spatial_gradient[ispin_left](1, 2)) / 4 /
        //     delta[0]); gradient[ispin][1] -= 2*exchange_tensor(2, 0) / Ms * ((spatial_gradient[ispin_top](1, 0) -
        //     spatial_gradient[ispin_bottom](1, 0)) / 4 / delta[2] + (spatial_gradient[ispin_right](1, 2) -
        //     spatial_gradient[ispin_left](1, 2)) / 4 / delta[0]); gradient[ispin][2] -= 2*exchange_tensor(0, 2) / Ms *
        //     ((spatial_gradient[ispin_top](2, 0) - spatial_gradient[ispin_bottom](2, 0)) / 4 / delta[2] +
        //     (spatial_gradient[ispin_right](2, 2) - spatial_gradient[ispin_left](2, 2)) / 4 / delta[0]);
        //     gradient[ispin][2]
        //     -= 2*exchange_tensor(2, 0) / Ms * ((spatial_gradient[ispin_top](2, 0) - spatial_gradient[ispin_bottom](2,
        //     0)) / 4 / delta[2] + (spatial_gradient[ispin_right](2, 2) - spatial_gradient[ispin_left](2, 2)) / 4 /
        //     delta[0]);

        //     // yz
        //     ispin_right  = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms,
        //     geometry->atom_types, neigh[2]); ispin_left   = idx_from_pair(ispin, boundary_conditions,
        //     geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, neigh[3]); ispin_top    =
        //     idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms,
        //     geometry->atom_types, neigh[4]); ispin_bottom = idx_from_pair(ispin, boundary_conditions,
        //     geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, neigh[5]);

        //     if( ispin_right == -1 )
        //         ispin_right = ispin;
        //     if( ispin_left == -1 )
        //         ispin_left = ispin;
        //     if( ispin_top == -1 )
        //         ispin_top = ispin;
        //     if( ispin_bottom == -1 )
        //         ispin_bottom = ispin;

        //     gradient[ispin][0] -= 2 * exchange_tensor(1, 2) / Ms * ((spatial_gradient[ispin_top](0, 1) -
        //     spatial_gradient[ispin_bottom](0, 1)) / 4 / delta[2] + (spatial_gradient[ispin_right](0, 2) -
        //     spatial_gradient[ispin_left](0, 2)) / 4 / delta[0]); gradient[ispin][0] -= 2 * exchange_tensor(2, 1) / Ms
        //     * ((spatial_gradient[ispin_top](0, 1) - spatial_gradient[ispin_bottom](0, 1)) / 4 / delta[2] +
        //     (spatial_gradient[ispin_right](0, 2) - spatial_gradient[ispin_left](0, 2)) / 4 / delta[0]);
        //     gradient[ispin][1]
        //     -= 2 * exchange_tensor(1, 2) / Ms * ((spatial_gradient[ispin_top](1, 1) -
        //     spatial_gradient[ispin_bottom](1, 1)) / 4 / delta[2] + (spatial_gradient[ispin_right](1, 2) -
        //     spatial_gradient[ispin_left](1, 2)) / 4 / delta[0]); gradient[ispin][1] -= 2 * exchange_tensor(2, 1) / Ms
        //     * ((spatial_gradient[ispin_top](1, 1) - spatial_gradient[ispin_bottom](1, 1)) / 4 / delta[2] +
        //     (spatial_gradient[ispin_right](1, 2) - spatial_gradient[ispin_left](1, 2)) / 4 / delta[0]);
        //     gradient[ispin][2]
        //     -= 2 * exchange_tensor(1, 2) / Ms * ((spatial_gradient[ispin_top](2, 1) -
        //     spatial_gradient[ispin_bottom](2, 1)) / 4 / delta[2] + (spatial_gradient[ispin_right](2, 2) -
        //     spatial_gradient[ispin_left](2, 2)) / 4 / delta[0]); gradient[ispin][2] -= 2 * exchange_tensor(2, 1) / Ms
        //     * ((spatial_gradient[ispin_top](2, 1) - spatial_gradient[ispin_bottom](2, 1)) / 4 / delta[2] +
        //     (spatial_gradient[ispin_right](2, 2) - spatial_gradient[ispin_left](2, 2)) / 4 / delta[0]);

        // }
    }
}

void Hamiltonian_Micromagnetic::Spatial_Gradient( const vectorfield & spins )
{
    auto & delta = geometry->cell_size;
// scalar delta[3] = { 3e-10,3e-10,3e-9 };
// scalar delta[3] = { 277e-12, 277e-12, 277e-12 };
/*
dn1/dr1 dn1/dr2 dn1/dr3
dn2/dr1 dn2/dr2 dn2/dr3
dn3/dr1 dn3/dr2 dn3/dr3
*/
#pragma omp parallel for
    for( unsigned int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        spatial_gradient[icell] = Matrix3::Zero();
        for( unsigned int i = 0; i < 3; ++i )
        {
            int icell_plus = idx_from_pair(
                icell, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                neigh[2 * i] );
            int icell_minus = idx_from_pair(
                icell, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                neigh[2 * i + 1] );

            if( icell_plus >= 0 || icell_minus >= 0 )
            {
                if( icell_plus == -1 )
                    icell_plus = icell;
                if( icell_minus == -1 )
                    icell_minus = icell;

                spatial_gradient[icell].col( i ) += ( spins[icell_plus] - spins[icell_minus] ) / ( 2 * delta[i] );
            }
        }
    }
}

void Hamiltonian_Micromagnetic::Gradient_DMI( const vectorfield & spins, vectorfield & gradient )
{
#pragma omp parallel for
    for( unsigned int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        for( unsigned int i = 0; i < 3; ++i )
        {
            gradient[icell][0] -= 4 * C::mu_B
                                  * ( dmi_tensor( 1, i ) * spatial_gradient[icell]( 2, i )
                                      - dmi_tensor( 2, i ) * spatial_gradient[icell]( 1, i ) )
                                  / Ms;
            gradient[icell][1] -= 4 * C::mu_B
                                  * ( dmi_tensor( 2, i ) * spatial_gradient[icell]( 0, i )
                                      - dmi_tensor( 0, i ) * spatial_gradient[icell]( 2, i ) )
                                  / Ms;
            gradient[icell][2] -= 4 * C::mu_B
                                  * ( dmi_tensor( 0, i ) * spatial_gradient[icell]( 1, i )
                                      - dmi_tensor( 1, i ) * spatial_gradient[icell]( 0, i ) )
                                  / Ms;
        }
    }
}

void Hamiltonian_Micromagnetic::Gradient_DDI( const vectorfield & spins, vectorfield & gradient )
{
    if( this->ddi_method == DDI_Method::FFT )
        this->Gradient_DDI_FFT( spins, gradient );
}

void Hamiltonian_Micromagnetic::Gradient_DDI_FFT( const vectorfield & spins, vectorfield & gradient )
{
    // Size of original geometry
    int Na = geometry->n_cells[0];
    int Nb = geometry->n_cells[1];
    int Nc = geometry->n_cells[2];

    auto cell_volume = geometry->cell_size[0] * geometry->cell_size[1] * geometry->cell_size[2];

    FFT_Spins( spins );

    auto & ft_D_matrices = transformed_dipole_matrices;
    auto & ft_spins      = fft_plan_spins.cpx_ptr;

    auto & res_iFFT = fft_plan_reverse.real_ptr;
    auto & res_mult = fft_plan_reverse.cpx_ptr;

    int idx_s, idx_d;

    // Workaround for compability with intel compiler
    const int c_n_cell_atoms               = geometry->n_cell_atoms;
    const int * c_it_bounds_pointwise_mult = it_bounds_pointwise_mult.data();

// Loop over basis atoms (i.e sublattices)
#pragma omp parallel for collapse( 3 )

    for( int c = 0; c < c_it_bounds_pointwise_mult[2]; ++c )
    {
        for( int b = 0; b < c_it_bounds_pointwise_mult[1]; ++b )
        {
            for( int a = 0; a < c_it_bounds_pointwise_mult[0]; ++a )
            {
                idx_s = a * spin_stride.a + b * spin_stride.b + c * spin_stride.c;
                idx_d = a * dipole_stride.a + b * dipole_stride.b + c * dipole_stride.c;

                auto & fs_x = ft_spins[idx_s];
                auto & fs_y = ft_spins[idx_s + 1 * spin_stride.comp];
                auto & fs_z = ft_spins[idx_s + 2 * spin_stride.comp];

                auto & fD_xx = ft_D_matrices[idx_d];
                auto & fD_xy = ft_D_matrices[idx_d + 1 * dipole_stride.comp];
                auto & fD_xz = ft_D_matrices[idx_d + 2 * dipole_stride.comp];
                auto & fD_yy = ft_D_matrices[idx_d + 3 * dipole_stride.comp];
                auto & fD_yz = ft_D_matrices[idx_d + 4 * dipole_stride.comp];
                auto & fD_zz = ft_D_matrices[idx_d + 5 * dipole_stride.comp];

                FFT::addTo(
                    res_mult[idx_s + 0 * spin_stride.comp], FFT::mult3D( fD_xx, fD_xy, fD_xz, fs_x, fs_y, fs_z ),
                    true );
                FFT::addTo(
                    res_mult[idx_s + 1 * spin_stride.comp], FFT::mult3D( fD_xy, fD_yy, fD_yz, fs_x, fs_y, fs_z ),
                    true );
                FFT::addTo(
                    res_mult[idx_s + 2 * spin_stride.comp], FFT::mult3D( fD_xz, fD_yz, fD_zz, fs_x, fs_y, fs_z ),
                    true );
            }
        }
    } // end iteration over padded lattice cells

    // Inverse Fourier Transform
    FFT::batch_iFour_3D( fft_plan_reverse );

    // Workaround for compability with intel compiler
    const int * c_n_cells = geometry->n_cells.data();

    // Place the gradients at the correct positions and mult with correct mu
    for( int c = 0; c < c_n_cells[2]; ++c )
    {
        for( int b = 0; b < c_n_cells[1]; ++b )
        {
            for( int a = 0; a < c_n_cells[0]; ++a )
            {
                int idx_orig = a + Na * ( b + Nb * c );
                int idx      = a * spin_stride.a + b * spin_stride.b + c * spin_stride.c;
                gradient[idx_orig][0] -= res_iFFT[idx] / sublattice_size;
                gradient[idx_orig][1] -= res_iFFT[idx + 1 * spin_stride.comp] / sublattice_size;
                gradient[idx_orig][2] -= res_iFFT[idx + 2 * spin_stride.comp] / sublattice_size;
            }
        }
    } // end iteration sublattice 1
}

void Hamiltonian_Micromagnetic::E_DDI_FFT( const vectorfield & spins, scalarfield & Energy )
{
    scalar Energy_DDI = 0;
    vectorfield gradients_temp;
    gradients_temp.resize( geometry->nos );
    Vectormath::fill( gradients_temp, { 0, 0, 0 } );
    this->Gradient_DDI_FFT( spins, gradients_temp );

    // === DEBUG: begin gradient comparison ===
    // vectorfield gradients_temp_dir;
    // gradients_temp_dir.resize(this->geometry->nos);
    // Vectormath::fill(gradients_temp_dir, {0,0,0});
    // Gradient_DDI_Direct(spins, gradients_temp_dir);

    // //get deviation
    // std::array<scalar, 3> deviation = {0,0,0};
    // std::array<scalar, 3> avg = {0,0,0};
    // for(int i = 0; i < this->geometry->nos; i++)
    // {
    //     for(int d = 0; d < 3; d++)
    //     {
    //         deviation[d] += std::pow(gradients_temp[i][d] - gradients_temp_dir[i][d], 2);
    //         avg[d] += gradients_temp_dir[i][d];
    //     }
    // }
    // std::cerr << "Avg. Gradient = " << avg[0]/this->geometry->nos << " " << avg[1]/this->geometry->nos << " " <<
    // avg[2]/this->geometry->nos << std::endl; std::cerr << "Avg. Deviation = " << deviation[0]/this->geometry->nos <<
    // " " << deviation[1]/this->geometry->nos << " " << deviation[2]/this->geometry->nos << std::endl;
//==== DEBUG: end gradient comparison ====

// TODO: add dot_scaled to Vectormath and use that
#pragma omp parallel for
    for( int ispin = 0; ispin < geometry->nos; ispin++ )
    {
        Energy[ispin] += 0.5 * spins[ispin].dot( gradients_temp[ispin] );
        // Energy_DDI    += 0.5 * spins[ispin].dot(gradients_temp[ispin]);
    }
}

void Hamiltonian_Micromagnetic::FFT_Demag_Tensors( FFT::FFT_Plan & fft_plan_dipole, int img_a, int img_b, int img_c )
{

    auto delta       = geometry->cell_size;
    auto cell_volume = geometry->cell_size[0] * geometry->cell_size[1] * geometry->cell_size[2];

    // Prefactor of DDI
    // The energy is proportional to  spin_direction * Demag_tensor * spin_direction
    // The 'mult' factor is chosen such that the cell resolved energy has
    // the dimension of total energy per cell in meV

    scalar mult = C::mu_0 / cell_volume * ( cell_volume * Ms * C::Joule ) * ( cell_volume * Ms * C::Joule );

    // Size of original geometry
    int Na = geometry->n_cells[0];
    int Nb = geometry->n_cells[1];
    int Nc = geometry->n_cells[2];

    auto & fft_dipole_inputs = fft_plan_dipole.real_ptr;

    // Iterate over the padded system
    const int * c_n_cells_padded = n_cells_padded.data();

#pragma omp parallel for collapse( 3 )
    for( int c = 0; c < c_n_cells_padded[2]; ++c )
    {
        for( int b = 0; b < c_n_cells_padded[1]; ++b )
        {
            for( int a = 0; a < c_n_cells_padded[0]; ++a )
            {
                int a_idx = a < Na ? a : a - n_cells_padded[0];
                int b_idx = b < Nb ? b : b - n_cells_padded[1];
                int c_idx = c < Nc ? c : c - n_cells_padded[2];

                scalar Dxx = 0, Dxy = 0, Dxz = 0, Dyy = 0, Dyz = 0, Dzz = 0;

                // Iterate over periodic images
                for( int a_pb = -img_a; a_pb <= img_a; a_pb++ )
                {
                    for( int b_pb = -img_b; b_pb <= img_b; b_pb++ )
                    {
                        for( int c_pb = -img_c; c_pb <= img_c; c_pb++ )
                        {
                            scalar X  = ( a_idx + a_pb * Na ) * delta[0];
                            scalar Y  = ( b_idx + b_pb * Nb ) * delta[1];
                            scalar Z  = ( c_idx + c_pb * Nc ) * delta[2];
                            scalar dx = delta[0];
                            scalar dy = delta[1];
                            scalar dz = delta[2];

                            Dxx += mult * Demagnetization_Tensor::Automatic::Nxx( X, Y, Z, dx, dy, dz );
                            Dxy += mult * Demagnetization_Tensor::Automatic::Nxy( X, Y, Z, dx, dy, dz );
                            Dxz += mult * Demagnetization_Tensor::Automatic::Nxy( X, Z, Y, dx, dz, dy );
                            Dyy += mult * Demagnetization_Tensor::Automatic::Nxx( Y, X, Z, dy, dx, dz );
                            Dyz += mult * Demagnetization_Tensor::Automatic::Nxy( Z, Y, X, dz, dy, dx );
                            Dzz += mult * Demagnetization_Tensor::Automatic::Nxx( Z, Y, X, dz, dy, dx );
                        }
                    }
                }

                int idx = a * dipole_stride.a + b * dipole_stride.b + c * dipole_stride.c;

                fft_dipole_inputs[idx]                          = Dxx;
                fft_dipole_inputs[idx + 1 * dipole_stride.comp] = Dxy;
                fft_dipole_inputs[idx + 2 * dipole_stride.comp] = Dxz;
                fft_dipole_inputs[idx + 3 * dipole_stride.comp] = Dyy;
                fft_dipole_inputs[idx + 4 * dipole_stride.comp] = Dyz;
                fft_dipole_inputs[idx + 5 * dipole_stride.comp] = Dzz;
            }
        }
    }
    FFT::batch_Four_3D( fft_plan_dipole );
}

void Hamiltonian_Micromagnetic::FFT_Spins( const vectorfield & spins )
{
    // size of original geometry
    int Na = geometry->n_cells[0];
    int Nb = geometry->n_cells[1];
    int Nc = geometry->n_cells[2];

    auto cell_volume = geometry->cell_size[0] * geometry->cell_size[1] * geometry->cell_size[2];

    auto & fft_spin_inputs = fft_plan_spins.real_ptr;

// iterate over the **original** system
#pragma omp parallel for collapse( 4 )
    for( int c = 0; c < Nc; ++c )
    {
        for( int b = 0; b < Nb; ++b )
        {
            for( int a = 0; a < Na; ++a )
            {
                int idx_orig = a + Na * ( b + Nb * c );
                int idx      = a * spin_stride.a + b * spin_stride.b + c * spin_stride.c;

                fft_spin_inputs[idx]                        = spins[idx_orig][0];
                fft_spin_inputs[idx + 1 * spin_stride.comp] = spins[idx_orig][1];
                fft_spin_inputs[idx + 2 * spin_stride.comp] = spins[idx_orig][2];
            }
        }
    }

    FFT::batch_Four_3D( fft_plan_spins );
}

void Hamiltonian_Micromagnetic::Prepare_DDI()
{
    Clean_DDI();

    if( ddi_method != DDI_Method::FFT )
        return;

    // We perform zero-padding in a lattice direction if the dimension of the system is greater than 1 *and*
    //  - the boundary conditions are open, or
    //  - the boundary conditions are periodic and zero-padding is explicitly requested
    n_cells_padded.resize( 3 );
    for( int i = 0; i < 3; i++ )
    {
        n_cells_padded[i]         = geometry->n_cells[i];
        bool perform_zero_padding = geometry->n_cells[i] > 1 && ( boundary_conditions[i] == 0 || ddi_pb_zero_padding );
        if( perform_zero_padding )
            n_cells_padded[i] *= 2;
    }
    sublattice_size = n_cells_padded[0] * n_cells_padded[1] * n_cells_padded[2];

    FFT::FFT_Init();

// Workaround for bug in kissfft
// kissfft_ndr does not perform one-dimensional FFTs properly
#ifndef SPIRIT_USE_FFTW
    int number_of_one_dims = 0;
    for( int i = 0; i < 3; i++ )
        if( n_cells_padded[i] == 1 && ++number_of_one_dims > 1 )
            n_cells_padded[i] = 2;
#endif

    sublattice_size = n_cells_padded[0] * n_cells_padded[1] * n_cells_padded[2];

    // We dont need to transform over length 1 dims
    std::vector<int> fft_dims;
    for( int i = 2; i >= 0; i-- ) // notice that reverse order is important!
    {
        if( n_cells_padded[i] > 1 )
            fft_dims.push_back( n_cells_padded[i] );
    }

    // Create FFT plans
    FFT::FFT_Plan fft_plan_dipole = FFT::FFT_Plan( fft_dims, false, 6, sublattice_size );
    fft_plan_spins                = FFT::FFT_Plan( fft_dims, false, 3, sublattice_size );
    fft_plan_reverse              = FFT::FFT_Plan( fft_dims, true, 3, sublattice_size );

#ifdef SPIRIT_USE_FFTW
    field<int *> temp_s = { &spin_stride.comp, &spin_stride.basis, &spin_stride.a, &spin_stride.b, &spin_stride.c };
    field<int *> temp_d
        = { &dipole_stride.comp, &dipole_stride.basis, &dipole_stride.a, &dipole_stride.b, &dipole_stride.c };

    FFT::get_strides(
        temp_s, { 3, this->geometry->n_cell_atoms, n_cells_padded[0], n_cells_padded[1], n_cells_padded[2] } );
    FFT::get_strides( temp_d, { 6, 1, n_cells_padded[0], n_cells_padded[1], n_cells_padded[2] } );
    it_bounds_pointwise_mult = { ( n_cells_padded[0] / 2 + 1 ), // due to redundancy in real fft
                                 n_cells_padded[1], n_cells_padded[2] };
#else
    field<int *> temp_s = { &spin_stride.a, &spin_stride.b, &spin_stride.c, &spin_stride.comp, &spin_stride.basis };
    field<int *> temp_d
        = { &dipole_stride.a, &dipole_stride.b, &dipole_stride.c, &dipole_stride.comp, &dipole_stride.basis };

    FFT::get_strides(
        temp_s, { n_cells_padded[0], n_cells_padded[1], n_cells_padded[2], 3, this->geometry->n_cell_atoms } );
    FFT::get_strides( temp_d, { n_cells_padded[0], n_cells_padded[1], n_cells_padded[2], 6, 1 } );
    it_bounds_pointwise_mult = { n_cells_padded[0], n_cells_padded[1], n_cells_padded[2] };
    ( it_bounds_pointwise_mult[fft_dims.size() - 1] /= 2 )++;
#endif

    // Perform FFT of dipole matrices
    int img_a = boundary_conditions[0] == 0 ? 0 : ddi_n_periodic_images[0];
    int img_b = boundary_conditions[1] == 0 ? 0 : ddi_n_periodic_images[1];
    int img_c = boundary_conditions[2] == 0 ? 0 : ddi_n_periodic_images[2];

    FFT_Demag_Tensors( fft_plan_dipole, img_a, img_b, img_c );
    transformed_dipole_matrices = std::move( fft_plan_dipole.cpx_ptr );
}

void Hamiltonian_Micromagnetic::Clean_DDI()
{
    fft_plan_spins   = FFT::FFT_Plan();
    fft_plan_reverse = FFT::FFT_Plan();
}

void Hamiltonian_Micromagnetic::Hessian( const vectorfield & spins, MatrixX & hessian ) {}

// Hamiltonian name as string
static const std::string name = "Micromagnetic";
const std::string & Hamiltonian_Micromagnetic::Name()
{
    return name;
}

} // namespace Engine

#endif