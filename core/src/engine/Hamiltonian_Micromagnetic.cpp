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

#include <fmt/format.h>

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
        : Ms( Ms ),
          Hamiltonian( boundary_conditions ),
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
    neigh.push_back( { 0, 0, { 1, 0, 0 } } );
    neigh.push_back( { 0, 0, { -1, 0, 0 } } );
    neigh.push_back( { 0, 0, { 0, 1, 0 } } );
    neigh.push_back( { 0, 0, { 0, -1, 0 } } );
    neigh.push_back( { 0, 0, { 0, 0, 1 } } );
    neigh.push_back( { 0, 0, { 0, 0, -1 } } );

    this->spatial_gradient = field<Matrix3>( geometry->nos, Matrix3::Zero() );
    this->Prepare_DDI();
    this->Update_Energy_Contributions();
}

void Hamiltonian_Micromagnetic::Update_Energy_Contributions()
{
    this->energy_contributions_per_spin = std::vector<std::pair<std::string, scalarfield>>( 0 );

    // External field
    if( std::abs( this->external_field_magnitude ) > 0 )
    {
        this->energy_contributions_per_spin.push_back( { "Zeeman", scalarfield( 0 ) } );
        this->idx_zeeman = this->energy_contributions_per_spin.size() - 1;
    }
    else
        this->idx_zeeman = -1;

    if( anisotropy_tensor.norm() > 0.0 )
    {
        this->energy_contributions_per_spin.push_back( { "Anisotropy", scalarfield( 0 ) } );
        this->idx_anisotropy = this->energy_contributions_per_spin.size() - 1;
    }
    else
        this->idx_anisotropy = -1;

    if( exchange_tensor.norm() > 0.0 )
    {
        this->energy_contributions_per_spin.push_back( { "Exchange", scalarfield( 0 ) } );
        this->idx_exchange = this->energy_contributions_per_spin.size() - 1;
    }
    else
        this->idx_exchange = -1;

    if( dmi_tensor.norm() > 0.0 )
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

    // printf("idx_zeeman %i\n", idx_zeeman);
    // printf("idx_exchange %i\n", idx_exchange);
    // printf("idx_dmi %i\n", idx_dmi);
    // printf("idx_anisotropy %i\n", idx_anisotropy);
    // printf("idx_ddi %i\n", idx_ddi);
    // std::cout << exchange_tensor << "\n";
    // std::cout << dmi_tensor << "\n ===== \n";
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
#pragma omp parallel for
    for( int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        if( check_atom_type( this->geometry->atom_types[icell] ) )
            Energy[icell] -= C::Joule * geometry->cell_volume * Ms * this->external_field_magnitude
                             * this->external_field_normal.dot( spins[icell] );
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

void Hamiltonian_Micromagnetic::E_Anisotropy( const vectorfield & spins, scalarfield & Energy )
{
#pragma omp parallel for
    for( int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        Energy[icell] -= geometry->cell_volume * C::Joule * spins[icell].dot( anisotropy_tensor * spins[icell] );
    }
}

void Hamiltonian_Micromagnetic::E_Exchange( const vectorfield & spins, scalarfield & Energy )
{
    auto delta = geometry->cell_size;
    for( unsigned int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        Vector3 grad_n;
        for( unsigned int alpha = 0; alpha < 3; ++alpha )
        {
            int icell_plus = idx_from_pair(
                icell, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                neigh[2 * alpha] );

            int icell_minus = idx_from_pair(
                icell, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                neigh[2 * alpha + 1] );

            if( icell_plus >= 0 || icell_minus >= 0 )
            {
                if( icell_plus == -1 )
                    icell_plus = icell;
                if( icell_minus == -1 )
                    icell_minus = icell;

                grad_n = ( spins[icell_plus] - spins[icell_minus] ) / ( 2 * delta[alpha] );
                // meV/J * J/m * 1/m * 1/m * m^3
                Energy[icell] += C::Joule * geometry->cell_volume * ( grad_n.dot( exchange_tensor * grad_n ) );
            }
        }
    }
}

void Hamiltonian_Micromagnetic::E_DMI( const vectorfield & spins, scalarfield & Energy )
{
    // TODO: This implementation is very likely far from optimal (performance wise)
    const auto & delta = geometry->cell_size;

    auto epsilon = []( int i, int j, int k ) { return -0.5 * ( j - i ) * ( k - j ) * ( i - k ); };

    scalar mult = C::Joule * geometry->cell_volume;

// Implements: epsilon_{mu,alpha,beta} * D_{mu, nu} * [ n_{alpha} dn_{beta}/dr_{nu} - n_{beta} dn_{alpha}/dr_{nu} ]
#pragma omp parallel for
    for( int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        for( int nu = 0; nu < 3; ++nu )
        {
            int icell_plus = idx_from_pair(
                icell, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                neigh[2 * nu] );

            int icell_minus = idx_from_pair(
                icell, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                neigh[2 * nu + 1] );

            if( icell_plus >= 0 || icell_minus >= 0 )
            {
                if( icell_plus == -1 )
                    icell_plus = icell;
                if( icell_minus == -1 )
                    icell_minus = icell;

                // Todo: Why is there a factor of 2 difference to OOMMF?
                Vector3 grad_n = ( spins[icell_plus] - spins[icell_minus] ) / ( 2 * delta[nu] );

                for( int alpha = 0; alpha < 3; alpha++ )
                {
                    for( int beta = 0; beta < 3; beta++ )
                    {
                        for( int mu = 0; mu < 3; mu++ )
                        {
                            // meV/J * J/m^2 * 1/m
                            Energy[icell]
                                += mult * epsilon( mu, alpha, beta ) * dmi_tensor( mu, nu )
                                   * ( spins[icell][alpha] * grad_n[beta] - spins[icell][beta] * grad_n[alpha] );
                        }
                    }
                }
            }
        }
    }
}

void Hamiltonian_Micromagnetic::E_DDI( const vectorfield & spins, scalarfield & Energy )
{
    if( this->ddi_method == DDI_Method::FFT )
        this->E_DDI_FFT( spins, Energy );
    else if( this->ddi_method == DDI_Method::Cutoff )
    {
        if( ddi_cutoff_radius < 0 )
            this->E_DDI_Direct( spins, Energy );
    }
}

void Hamiltonian_Micromagnetic::E_DDI_Direct( const vectorfield & spins, scalarfield & Energy )
{
    vectorfield gradients_temp;
    gradients_temp.resize( geometry->nos );
    Vectormath::fill( gradients_temp, { 0, 0, 0 } );
    this->Gradient_DDI_Direct( spins, gradients_temp );

#pragma omp parallel for
    for( int ispin = 0; ispin < geometry->nos; ispin++ )
    {
        Energy[ispin] += 0.5 * spins[ispin].dot( gradients_temp[ispin] );
    }
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

#pragma omp parallel for
    for( int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        if( check_atom_type( this->geometry->atom_types[icell] ) )
            gradient[icell]
                -= C::Joule * Ms * geometry->cell_volume * this->external_field_magnitude * this->external_field_normal;
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
        gradient[icell] -= 2.0 * geometry->cell_volume * C::Joule * anisotropy_tensor * spins[icell];
    }
}

void Hamiltonian_Micromagnetic::Gradient_Exchange( const vectorfield & spins, vectorfield & gradient )
{
    auto & delta = geometry->cell_size;

#pragma omp parallel for
    for( unsigned int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        for( int alpha = 0; alpha < 3; ++alpha )
        {
            int icell_plus = idx_from_pair(
                icell, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                neigh[2 * alpha] );

            int icell_minus = idx_from_pair(
                icell, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                neigh[2 * alpha + 1] );

            if( icell_plus >= 0 )
            {
                Vector3 grad_n = spatial_gradient[icell_plus].col( alpha );
                gradient[icell]
                    -= C::Joule * geometry->cell_volume * 2 * ( exchange_tensor * grad_n ) / ( 2 * delta[alpha] );
            }
            else
            {
                Vector3 grad_n = spatial_gradient[icell].col( alpha );
                gradient[icell]
                    += C::Joule * geometry->cell_volume * 2 * ( exchange_tensor * grad_n ) / ( 2 * delta[alpha] );
            }

            if( icell_minus >= 0 )
            {
                Vector3 grad_n = spatial_gradient[icell_minus].col( alpha );
                gradient[icell]
                    += C::Joule * geometry->cell_volume * 2 * ( exchange_tensor * grad_n ) / ( 2 * delta[alpha] );
            }
            else
            {
                Vector3 grad_n = spatial_gradient[icell].col( alpha );
                gradient[icell]
                    -= C::Joule * geometry->cell_volume * 2 * ( exchange_tensor * grad_n ) / ( 2 * delta[alpha] );
            }
        }
    }
}

void Hamiltonian_Micromagnetic::Spatial_Gradient( const vectorfield & spins )
{
    auto & delta = geometry->cell_size;

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
    const auto & delta = geometry->cell_size;

    auto epsilon = []( int i, int j, int k ) { return -0.5 * ( j - i ) * ( k - j ) * ( i - k ); };

    scalar mult = C::Joule * geometry->cell_volume;

#pragma omp parallel for
    for( unsigned int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        for( unsigned int nu = 0; nu < 3; ++nu )
        {
            int icell_plus = idx_from_pair(
                icell, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                neigh[2 * nu] );

            int icell_minus = idx_from_pair(
                icell, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                neigh[2 * nu + 1] );

            // Energy[icell] += mult * epsilon(mu, alpha, beta) * dmi_tensor(mu, nu) * ( spins[icell][alpha] *
            // grad_n[beta] - spins[icell][beta] * grad_n[alpha] );

            Vector3 grad_n = spatial_gradient[icell].col( nu );

            for( int alpha = 0; alpha < 3; alpha++ )
            {
                for( int beta = 0; beta < 3; beta++ )
                {
                    for( int mu = 0; mu < 3; mu++ )
                    {
                        if( icell_plus >= 0 )
                        {
                            gradient[icell][beta] -= mult * epsilon( mu, alpha, beta ) * dmi_tensor( mu, nu )
                                                     * spins[icell_plus][alpha] / ( 2 * delta[nu] );
                            gradient[icell][alpha] += mult * epsilon( mu, alpha, beta ) * dmi_tensor( mu, nu )
                                                      * spins[icell_plus][beta] / ( 2 * delta[nu] );
                        }

                        if( icell_minus >= 0 )
                        {
                            gradient[icell][beta] += mult * epsilon( mu, alpha, beta ) * dmi_tensor( mu, nu )
                                                     * spins[icell_minus][alpha] / ( 2 * delta[nu] );
                            gradient[icell][alpha] -= mult * epsilon( mu, alpha, beta ) * dmi_tensor( mu, nu )
                                                      * spins[icell_minus][beta] / ( 2 * delta[nu] );
                        }

                        gradient[icell][alpha]
                            += mult * epsilon( mu, alpha, beta ) * dmi_tensor( mu, nu ) * ( grad_n[beta] );
                        if( icell_plus < 0 )
                        {
                            gradient[icell][beta] += mult * epsilon( mu, alpha, beta ) * dmi_tensor( mu, nu )
                                                     * spins[icell][alpha] / ( 2 * delta[nu] );
                            gradient[icell][alpha] -= mult * epsilon( mu, alpha, beta ) * dmi_tensor( mu, nu )
                                                      * spins[icell][beta] / ( 2 * delta[nu] );
                        }

                        gradient[icell][beta]
                            -= mult * epsilon( mu, alpha, beta ) * dmi_tensor( mu, nu ) * ( grad_n[alpha] );
                        if( icell_minus < 0 )
                        {
                            gradient[icell][beta] -= mult * epsilon( mu, alpha, beta ) * dmi_tensor( mu, nu )
                                                     * spins[icell][alpha] / ( 2 * delta[nu] );
                            gradient[icell][alpha] += mult * epsilon( mu, alpha, beta ) * dmi_tensor( mu, nu )
                                                      * spins[icell][beta] / ( 2 * delta[nu] );
                        }
                    }
                }
            }
        }
    }
}

void Hamiltonian_Micromagnetic::Gradient_DDI( const vectorfield & spins, vectorfield & gradient )
{
    if( this->ddi_method == DDI_Method::FFT )
        this->Gradient_DDI_FFT( spins, gradient );

    else if( this->ddi_method == DDI_Method::Cutoff )
    {
        if( ddi_cutoff_radius < 0 )
            this->Gradient_DDI_Direct( spins, gradient );
    }
}

void Hamiltonian_Micromagnetic::Gradient_DDI_Direct( const vectorfield & spins, vectorfield & gradient )
{
    Vector3 delta = geometry->cell_size;
    scalar mult   = Constants_Micromagnetic::mu_0 * geometry->cell_volume * ( Ms ) * (Ms)*C::Joule;

    int img_a = boundary_conditions[0] == 0 ? 0 : ddi_n_periodic_images[0];
    int img_b = boundary_conditions[1] == 0 ? 0 : ddi_n_periodic_images[1];
    int img_c = boundary_conditions[2] == 0 ? 0 : ddi_n_periodic_images[2];

    for( int idx1 = 0; idx1 < geometry->nos; idx1++ )
    {
        for( int idx2 = 0; idx2 < geometry->nos; idx2++ )
        {
            scalar Dxx = 0, Dxy = 0, Dxz = 0, Dyy = 0, Dyz = 0, Dzz = 0;
            auto diff = this->geometry->positions[idx2] - this->geometry->positions[idx1];

            for( int a_pb = -img_a; a_pb <= img_a; a_pb++ )
            {
                for( int b_pb = -img_b; b_pb <= img_b; b_pb++ )
                {
                    for( int c_pb = -img_c; c_pb <= img_c; c_pb++ )
                    {
                        scalar X = 1e-10 * diff[0] + geometry->n_cells[0] * a_pb * delta[0];
                        scalar Y = 1e-10 * diff[1] + geometry->n_cells[1] * b_pb * delta[1];
                        scalar Z = 1e-10 * diff[2] + geometry->n_cells[2] * c_pb * delta[2];

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

            gradient[idx1][0] -= ( Dxx * spins[idx2][0] + Dxy * spins[idx2][1] + Dxz * spins[idx2][2] );
            gradient[idx1][1] -= ( Dxy * spins[idx2][0] + Dyy * spins[idx2][1] + Dyz * spins[idx2][2] );
            gradient[idx1][2] -= ( Dxz * spins[idx2][0] + Dyz * spins[idx2][1] + Dzz * spins[idx2][2] );
        }
    }
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
    // Vector3 deviation = {0,0,0};
    // scalar max_deviation = 0;

    // std::array<scalar, 3> avg = {0,0,0};
    // for(int i = 0; i < this->geometry->nos; i++)
    // {
    //     for(int d = 0; d < 3; d++)
    //     {
    //         deviation[d] += std::pow(gradients_temp[i][d] - gradients_temp_dir[i][d], 2);
    //         avg[d]       += gradients_temp_dir[i][d];
    //     }
    //     max_deviation = std::max( (gradients_temp_dir[i] - gradients_temp[i]).norm(), max_deviation );
    // }
    // std::cerr << "Avg. Gradient = " << avg[0]/this->geometry->nos << " " << avg[1]/this->geometry->nos << " " <<
    // avg[2]/this->geometry->nos << std::endl; std::cerr << "Avg. Deviation = " << deviation[0]/this->geometry->nos
    // << " " << deviation[1]/this->geometry->nos << " " << deviation[2]/this->geometry->nos << std::endl;
    // std::cerr <<  "Max. Deviation = " << max_deviation << "\n";
    //==== DEBUG: end gradient comparison ====

    // TODO: add dot_scaled to Vectormath and use that
    for( int ispin = 0; ispin < geometry->nos; ispin++ )
    {
        Energy[ispin] += 0.5 * spins[ispin].dot( gradients_temp[ispin] );
        // Energy_DDI    += 0.5 * spins[ispin].dot(gradients_temp[ispin]);
    }
}

void Hamiltonian_Micromagnetic::FFT_Demag_Tensors( FFT::FFT_Plan & fft_plan_dipole, int img_a, int img_b, int img_c )
{
    auto delta = geometry->cell_size;

    // Prefactor of DDI
    // The energy is proportional to  spin_direction * Demag_tensor * spin_direction
    // The 'mult' factor is chosen such that the cell resolved energy has
    // the dimension of total energy per cell in meV

    // mult has the units of [N / A^2] [m^3] [(A/m)^2] [mev/J] = [J] [meV/J] = [meV]
    scalar mult = Constants_Micromagnetic::mu_0 * geometry->cell_volume * ( Ms ) * (Ms)*C::Joule;

    std::cout << "cell_size " << geometry->cell_size.transpose() << "\n";
    std::cout << "cell_volume " << geometry->cell_volume << "\n";
    std::cout << "mult " << mult << "\n";
    std::cout << "Ms   " << Ms << "\n";

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
                            scalar X = ( a_idx + a_pb * Na ) * delta[0];
                            scalar Y = ( b_idx + b_pb * Nb ) * delta[1];
                            scalar Z = ( c_idx + c_pb * Nc ) * delta[2];

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
#pragma omp parallel for collapse( 3 )
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