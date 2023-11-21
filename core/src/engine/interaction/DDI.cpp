#include <engine/Backend_par.hpp>
#include <engine/Indexing.hpp>
#include <engine/Neighbours.hpp>
#include <engine/Vectormath.hpp>
#include <engine/interaction/DDI.hpp>
#include <utility/Constants.hpp>
#include <utility>

#include <Eigen/Core>
#include <Eigen/Dense>

#ifndef SPIRIT_USE_CUDA
#include <algorithm>
#else
#include <complex> // TODO: check if I need complex for the CUDA implementation
#endif

using namespace Data;
using namespace Utility;
namespace C = Utility::Constants;
using Engine::Indexing::idx_from_pair;
using Engine::Indexing::idx_from_translations;

#ifdef SPIRIT_USE_CUDA
using Engine::Indexing::cu_check_atom_type;
using Engine::Indexing::cu_idx_from_pair;
using Engine::Indexing::cu_tupel_from_idx;
#endif

namespace Engine
{

namespace Interaction
{

DDI::DDI(
    Hamiltonian * hamiltonian, Engine::DDI_Method ddi_method, intfield n_periodic_images, const bool pb_zero_padding,
    const scalar cutoff_radius ) noexcept
        : Interaction::Base<DDI>( hamiltonian, scalarfield( 0 ) ),
          method( ddi_method ),
          ddi_n_periodic_images( std::move( n_periodic_images ) ),
          ddi_pb_zero_padding( pb_zero_padding ),
          ddi_cutoff_radius( cutoff_radius ),
          fft_plan_spins( FFT::FFT_Plan() ),
          fft_plan_reverse( FFT::FFT_Plan() )
{
    this->updateGeometry();
}

DDI::DDI( Hamiltonian * hamiltonian, Engine::DDI_Method ddi_method, const Data::DDI_Data & ddi_data ) noexcept
        : DDI( hamiltonian, ddi_method, ddi_data.n_periodic_images, ddi_data.pb_zero_padding, ddi_data.radius )
{
}

bool DDI::is_contributing() const
{
    return method != DDI_Method::None;
}

void DDI::updateFromGeometry( const Geometry * geometry )
{
    if( method == DDI_Method::Cutoff )
        ddi_pairs = Engine::Neighbours::Get_Pairs_in_Radius( *geometry, this->ddi_cutoff_radius );
    else
        ddi_pairs = field<Pair>{};

    ddi_magnitudes = scalarfield( ddi_pairs.size() );
    ddi_normals    = vectorfield( ddi_pairs.size() );

    for( std::size_t i = 0; i < this->ddi_pairs.size(); ++i )
    {
        Engine::Neighbours::DDI_from_Pair(
            *geometry,
            {
                this->ddi_pairs[i].i,
                this->ddi_pairs[i].j,
#ifndef SPIRIT_USE_CUDA
                this->ddi_pairs[i].translations,
#else
                { this->ddi_pairs[i].translations[0], this->ddi_pairs[i].translations[1],
                  this->ddi_pairs[i].translations[2] }
#endif
            },
            this->ddi_magnitudes[i], this->ddi_normals[i] );
    };
    // Dipole-dipole
    this->Prepare_DDI();
}

void DDI::Energy_per_Spin( const vectorfield & spins, scalarfield & energy )
{
    if( this->method == DDI_Method::FFT )
        this->Energy_per_Spin_FFT( spins, energy );
    else if( this->method == DDI_Method::Cutoff )
    {
        // TODO: Merge these implementations in the future
        if( ddi_cutoff_radius >= 0 )
            this->Energy_per_Spin_Cutoff( spins, energy );
        else
            this->Energy_per_Spin_Direct( spins, energy );
    }
};

void DDI::Hessian( const vectorfield & spins, MatrixX & hessian )
{
    const auto * geometry = hamiltonian->geometry.get();

    // Tentative Dipole-Dipole (only works for open boundary conditions)
    if( method != DDI_Method::None )
    {
        static constexpr scalar mult = C::mu_0 * C::mu_B * C::mu_B / ( 4 * C::Pi * 1e-30 );
        for( int idx1 = 0; idx1 < geometry->nos; idx1++ )
        {
            for( int idx2 = 0; idx2 < geometry->nos; idx2++ )
            {
                auto diff = geometry->positions[idx2] - geometry->positions[idx1];
                scalar d = diff.norm(), d3 = 0, d5 = 0;
                scalar Dxx = 0, Dxy = 0, Dxz = 0, Dyy = 0, Dyz = 0, Dzz = 0;
                if( d > 1e-10 )
                {
                    d3 = d * d * d;
                    d5 = d * d * d * d * d;
                    Dxx += mult * ( 3 * diff[0] * diff[0] / d5 - 1 / d3 );
                    Dxy += mult * 3 * diff[0] * diff[1] / d5; // same as Dyx
                    Dxz += mult * 3 * diff[0] * diff[2] / d5; // same as Dzx
                    Dyy += mult * ( 3 * diff[1] * diff[1] / d5 - 1 / d3 );
                    Dyz += mult * 3 * diff[1] * diff[2] / d5; // same as Dzy
                    Dzz += mult * ( 3 * diff[2] * diff[2] / d5 - 1 / d3 );
                }

                int i = 3 * idx1;
                int j = 3 * idx2;

                hessian( i + 0, j + 0 ) += -geometry->mu_s[idx1] * geometry->mu_s[idx2] * ( Dxx );
                hessian( i + 1, j + 0 ) += -geometry->mu_s[idx1] * geometry->mu_s[idx2] * ( Dxy );
                hessian( i + 2, j + 0 ) += -geometry->mu_s[idx1] * geometry->mu_s[idx2] * ( Dxz );
                hessian( i + 0, j + 1 ) += -geometry->mu_s[idx1] * geometry->mu_s[idx2] * ( Dxy );
                hessian( i + 1, j + 1 ) += -geometry->mu_s[idx1] * geometry->mu_s[idx2] * ( Dyy );
                hessian( i + 2, j + 1 ) += -geometry->mu_s[idx1] * geometry->mu_s[idx2] * ( Dyz );
                hessian( i + 0, j + 2 ) += -geometry->mu_s[idx1] * geometry->mu_s[idx2] * ( Dxz );
                hessian( i + 1, j + 2 ) += -geometry->mu_s[idx1] * geometry->mu_s[idx2] * ( Dyz );
                hessian( i + 2, j + 2 ) += -geometry->mu_s[idx1] * geometry->mu_s[idx2] * ( Dzz );
            }
        }
    }

    // // TODO: Dipole-Dipole
    // for (unsigned int i_pair = 0; i_pair < this->DD_indices.size(); ++i_pair)
    // {
    //     // indices
    //     int idx_1 = DD_indices[i_pair][0];
    //     int idx_2 = DD_indices[i_pair][1];
    //     // prefactor
    //     scalar prefactor = 0.0536814951168
    //         * mu_s[idx_1] * mu_s[idx_2]
    //         / std::pow(DD_magnitude[i_pair], 3);
    //     // components
    //     for (int alpha = 0; alpha < 3; ++alpha)
    //     {
    //         for (int beta = 0; beta < 3; ++beta)
    //         {
    //             int idx_h = idx_1 + alpha*nos + 3 * nos*(idx_2 + beta*nos);
    //             if (alpha == beta)
    //                 hessian[idx_h] += prefactor;
    //             hessian[idx_h] += -3.0*prefactor*DD_normal[i_pair][alpha] * DD_normal[i_pair][beta];
    //         }
    //     }
    // }
};
void DDI::Sparse_Hessian( const vectorfield & spins, std::vector<triplet> & hessian ){
    // TODO: Write a sparse Hessian implementation
};

void DDI::Gradient( const vectorfield & spins, vectorfield & gradient )
{
    if( this->method == DDI_Method::FFT )
        this->Gradient_FFT( spins, gradient );
    else if( this->method == DDI_Method::Cutoff )
    {
        // TODO: Merge these implementations in the future
        if( this->ddi_cutoff_radius >= 0 )
            this->Gradient_Cutoff( spins, gradient );
        else
            this->Gradient_Direct( spins, gradient );
    }
};

// Calculate the total energy for a single spin to be used in Monte Carlo.
//      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
scalar DDI::Energy_Single_Spin( const int ispin, const vectorfield & spins )
{
    // TODO
    return 0;
};

void DDI::Energy_per_Spin_Direct( const vectorfield & spins, scalarfield & energy )
{
    const auto * geometry = hamiltonian->geometry.get();

    vectorfield gradients_temp;
    gradients_temp.resize( geometry->nos );
    Vectormath::fill( gradients_temp, { 0, 0, 0 } );
    this->Gradient_Direct( spins, gradients_temp );

#pragma omp parallel for
    for( int ispin = 0; ispin < geometry->nos; ispin++ )
        energy[ispin] += 0.5 * spins[ispin].dot( gradients_temp[ispin] );
}

void DDI::Energy_per_Spin_Cutoff( const vectorfield & spins, scalarfield & energy )
{
    const auto * geometry            = hamiltonian->geometry.get();
    const auto & boundary_conditions = hamiltonian->boundary_conditions;

    const auto & mu_s = geometry->mu_s;
    // The translations are in angstr�m, so the |r|[m] becomes |r|[m]*10^-10
    static constexpr scalar mult = C::mu_0 * C::mu_B * C::mu_B / ( 4 * C::Pi * 1e-30 );

#ifdef SPIRIT_USE_CUDA
// //scalar mult = -mu_B*mu_B*1.0 / 4.0 / Pi; // multiply with mu_B^2
// scalar mult = 0.5*0.0536814951168; // mu_0*mu_B**2/(4pi*10**-30) -- the translations are in angstr�m, so the
// |r|[m] becomes |r|[m]*10^-10
// // scalar result = 0.0;

// for (unsigned int i_pair = 0; i_pair < ddi_pairs.size(); ++i_pair)
// {
//     if (ddi_magnitudes[i_pair] > 0.0)
//     {
//         for (int da = 0; da < geometry->n_cells[0]; ++da)
//         {
//             for (int db = 0; db < geometry->n_cells[1]; ++db)
//             {
//                 for (int dc = 0; dc < geometry->n_cells[2]; ++dc)
//                 {
//                     std::array<int, 3 > translations = { da, db, dc };
//                     // int idx_i = ddi_pairs[i_pair].i;
//                     // int idx_j = ddi_pairs[i_pair].j;
//                     int idx_i = idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations);
//                     int idx_j = idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations,
//                     ddi_pairs[i_pair].translations); Energy[idx_i] -= mult / std::pow(ddi_magnitudes[i_pair], 3.0) *
//                         (3 * spins[idx_j].dot(ddi_normals[i_pair]) * spins[idx_i].dot(ddi_normals[i_pair]) -
//                         spins[idx_i].dot(spins[idx_j]));
//                     energy[idx_j] -= mult / std::pow(ddi_magnitudes[i_pair], 3.0) *
//                         (3 * spins[idx_j].dot(ddi_normals[i_pair]) * spins[idx_i].dot(ddi_normals[i_pair]) -
//                         spins[idx_i].dot(spins[idx_j]));
//                 }
//             }
//         }
//     }
// }
#else

    for( unsigned int i_pair = 0; i_pair < ddi_pairs.size(); ++i_pair )
    {
        if( ddi_magnitudes[i_pair] > 0.0 )
        {
            for( int da = 0; da < geometry->n_cells[0]; ++da )
            {
                for( int db = 0; db < geometry->n_cells[1]; ++db )
                {
                    for( int dc = 0; dc < geometry->n_cells[2]; ++dc )
                    {
                        std::array<int, 3> translations = { da, db, dc };

                        int i = ddi_pairs[i_pair].i;
                        int j = ddi_pairs[i_pair].j;
                        int ispin
                            = i + idx_from_translations( geometry->n_cells, geometry->n_cell_atoms, translations );
                        int jspin = idx_from_pair(
                            ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                            ddi_pairs[i_pair] );
                        if( jspin >= 0 )
                        {
                            energy[ispin] -= 0.5 * mu_s[ispin] * mu_s[jspin] * mult
                                             / std::pow( ddi_magnitudes[i_pair], 3.0 )
                                             * ( 3 * spins[ispin].dot( ddi_normals[i_pair] )
                                                     * spins[jspin].dot( ddi_normals[i_pair] )
                                                 - spins[ispin].dot( spins[jspin] ) );
                        }
                    }
                }
            }
        }
    }
#endif
}

void DDI::Gradient_Cutoff( const vectorfield & spins, vectorfield & gradient )
{
    const auto * geometry            = hamiltonian->geometry.get();
    const auto & boundary_conditions = hamiltonian->boundary_conditions;
    const auto & mu_s                = geometry->mu_s;
#ifdef SPIRIT_USE_CUDA
// TODO
#else
    // The translations are in angstr�m, so the |r|[m] becomes |r|[m]*10^-10
    static constexpr scalar mult = C::mu_0 * C::mu_B * C::mu_B / ( 4 * C::Pi * 1e-30 );

    for( unsigned int i_pair = 0; i_pair < ddi_pairs.size(); ++i_pair )
    {
        if( ddi_magnitudes[i_pair] > 0.0 )
        {
            for( int da = 0; da < geometry->n_cells[0]; ++da )
            {
                for( int db = 0; db < geometry->n_cells[1]; ++db )
                {
                    for( int dc = 0; dc < geometry->n_cells[2]; ++dc )
                    {
                        scalar skalar_contrib           = mult / std::pow( ddi_magnitudes[i_pair], 3.0 );
                        std::array<int, 3> translations = { da, db, dc };

                        int i = ddi_pairs[i_pair].i;
                        int j = ddi_pairs[i_pair].j;
                        int ispin
                            = i + idx_from_translations( geometry->n_cells, geometry->n_cell_atoms, translations );
                        int jspin = idx_from_pair(
                            ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                            ddi_pairs[i_pair] );
                        if( jspin >= 0 )
                        {
                            gradient[ispin] -= mu_s[jspin] * mu_s[ispin] * skalar_contrib
                                               * ( 3 * ddi_normals[i_pair] * spins[jspin].dot( ddi_normals[i_pair] )
                                                   - spins[jspin] );
                        }
                    }
                }
            }
        }
    }
#endif
}

#ifdef SPIRIT_USE_CUDA
// TODO: add dot_scaled to Vectormath and use that
__global__ void CU_E_DDI_FFT(
    scalar * energy, const Vector3 * spins, const Vector3 * gradients, const int nos, const int n_cell_atoms,
    const scalar * mu_s )
{
    for( int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < nos; idx += blockDim.x * gridDim.x )
    {
        energy[idx] += 0.5 * spins[idx].dot( gradients[idx] );
    }
}
#endif

void DDI::Energy_per_Spin_FFT( const vectorfield & spins, scalarfield & energy )
{
    const auto * geometry = hamiltonian->geometry.get();

#ifdef SPIRIT_USE_CUDA
    // TODO: maybe the gradient should be cached somehow, it is quite inefficient to calculate it
    // again just for the energy
    vectorfield gradients_temp( geometry->nos );
    Vectormath::fill( gradients_temp, { 0, 0, 0 } );
    this->Gradient( spins, gradients_temp );
    CU_E_DDI_FFT<<<( geometry->nos + 1023 ) / 1024, 1024>>>(
        energy.data(), spins.data(), gradients_temp.data(), geometry->nos, geometry->n_cell_atoms,
        geometry->mu_s.data() );

    // === DEBUG: begin gradient comparison ===
    // vectorfield gradients_temp_dir;
    // gradients_temp_dir.resize(geometry->nos);
    // Vectormath::fill(gradients_temp_dir, {0,0,0});
    // Gradient_Direct(spins, gradients_temp_dir);

    // //get deviation
    // std::array<scalar, 3> deviation = {0,0,0};
    // std::array<scalar, 3> avg = {0,0,0};
    // std::array<scalar, 3> avg_ft = {0,0,0};

    // for(int i = 0; i < geometry->nos; i++)
    // {
    //     for(int d = 0; d < 3; d++)
    //     {
    //         deviation[d] += std::pow(gradients_temp[i][d] - gradients_temp_dir[i][d], 2);
    //         avg[d] += gradients_temp_dir[i][d];
    //         avg_ft[d] += gradients_temp[i][d];
    //     }
    // }
    // std::cerr << "Avg. Gradient (Direct) = " << avg[0]/geometry->nos << " " << avg[1]/geometry->nos << "
    // " << avg[2]/geometry->nos << std::endl; std::cerr << "Avg. Gradient (FFT)    = " <<
    // avg_ft[0]/geometry->nos << " " << avg_ft[1]/geometry->nos << " " << avg_ft[2]/geometry->nos <<
    // std::endl; std::cerr << "Relative Error in %    = " << (avg_ft[0]/avg[0]-1)*100 << " " <<
    // (avg_ft[1]/avg[1]-1)*100 << " " << (avg_ft[2]/avg[2]-1)*100 << std::endl; std::cerr << "Avg. Deviation         =
    // " << std::pow(deviation[0]/geometry->nos, 0.5) << " " << std::pow(deviation[1]/geometry->nos, 0.5) <<
    // " " << std::pow(deviation[2]/geometry->nos, 0.5) << std::endl; std::cerr << " ---------------- " <<
    // std::endl;
    // ==== DEBUG: end gradient comparison ====

#else
    // scalar Energy_DDI = 0;
    vectorfield gradients_temp;
    gradients_temp.resize( geometry->nos );
    Vectormath::fill( gradients_temp, { 0, 0, 0 } );
    this->Gradient_FFT( spins, gradients_temp );

// TODO: add dot_scaled to Vectormath and use that
#pragma omp parallel for
    for( int ispin = 0; ispin < geometry->nos; ispin++ )
    {
        energy[ispin] += 0.5 * spins[ispin].dot( gradients_temp[ispin] );
        // Energy_DDI    += 0.5 * spins[ispin].dot(gradients_temp[ispin]);
    }
#endif
}

#ifdef SPIRIT_USE_CUDA
__global__ void CU_FFT_Pointwise_Mult(
    FFT::FFT_cpx_type * ft_D_matrices, FFT::FFT_cpx_type * ft_spins, FFT::FFT_cpx_type * res_mult,
    int * iteration_bounds, int * inter_sublattice_lookup, FFT::StrideContainer dipole_stride,
    FFT::StrideContainer spin_stride )
{
    int nos = iteration_bounds[0] * iteration_bounds[1] * iteration_bounds[2] * iteration_bounds[3];
    int tupel[4];

    for( int ispin = blockIdx.x * blockDim.x + threadIdx.x; ispin < nos; ispin += blockDim.x * gridDim.x )
    {
        cu_tupel_from_idx( ispin, tupel, iteration_bounds, 4 ); // tupel now is {i_b1, a, b, c}
        int i_b1 = tupel[0], a = tupel[1], b = tupel[2], c = tupel[3];

        // Index to the first component of the spin (Remember: not the same as ispin, because we also have the spin
        // component stride!)
        int idx_b1 = i_b1 * spin_stride.basis + a * spin_stride.a + b * spin_stride.b + c * spin_stride.c;

        // Collect the intersublattice contributions
        FFT::FFT_cpx_type res_temp_x{ 0, 0 }, res_temp_y{ 0, 0 }, res_temp_z{ 0, 0 };
        for( int i_b2 = 0; i_b2 < iteration_bounds[0]; i_b2++ )
        {
            // Index to the first component of the second spin
            int idx_b2 = i_b2 * spin_stride.basis + a * spin_stride.a + b * spin_stride.b + c * spin_stride.c;

            int & b_inter = inter_sublattice_lookup[i_b1 + i_b2 * iteration_bounds[0]];
            // Index of the dipole matrix "connecting" the two spins
            int idx_d = b_inter * dipole_stride.basis + a * dipole_stride.a + b * dipole_stride.b + c * dipole_stride.c;

            // Fourier transformed components of the second spin
            auto & fs_x = ft_spins[idx_b2];
            auto & fs_y = ft_spins[idx_b2 + 1 * spin_stride.comp];
            auto & fs_z = ft_spins[idx_b2 + 2 * spin_stride.comp];

            // Fourier transformed components of the dipole matrix
            auto & fD_xx = ft_D_matrices[idx_d];
            auto & fD_xy = ft_D_matrices[idx_d + 1 * dipole_stride.comp];
            auto & fD_xz = ft_D_matrices[idx_d + 2 * dipole_stride.comp];
            auto & fD_yy = ft_D_matrices[idx_d + 3 * dipole_stride.comp];
            auto & fD_yz = ft_D_matrices[idx_d + 4 * dipole_stride.comp];
            auto & fD_zz = ft_D_matrices[idx_d + 5 * dipole_stride.comp];

            FFT::addTo( res_temp_x, FFT::mult3D( fD_xx, fD_xy, fD_xz, fs_x, fs_y, fs_z ) );
            FFT::addTo( res_temp_y, FFT::mult3D( fD_xy, fD_yy, fD_yz, fs_x, fs_y, fs_z ) );
            FFT::addTo( res_temp_z, FFT::mult3D( fD_xz, fD_yz, fD_zz, fs_x, fs_y, fs_z ) );
        }
        // Add the temporary result
        FFT::addTo( res_mult[idx_b1 + 0 * spin_stride.comp], res_temp_x, true );
        FFT::addTo( res_mult[idx_b1 + 1 * spin_stride.comp], res_temp_y, true );
        FFT::addTo( res_mult[idx_b1 + 2 * spin_stride.comp], res_temp_z, true );
    }
}

__global__ void CU_Write_FFT_Gradients(
    FFT::FFT_real_type * resiFFT, Vector3 * gradient, FFT::StrideContainer spin_stride, const int * iteration_bounds,
    const int n_cell_atoms, const scalar * mu_s, int sublattice_size )
{
    int nos = iteration_bounds[0] * iteration_bounds[1] * iteration_bounds[2] * iteration_bounds[3];
    int tupel[4];
    int idx_pad;

    for( int idx_orig = blockIdx.x * blockDim.x + threadIdx.x; idx_orig < nos; idx_orig += blockDim.x * gridDim.x )
    {
        cu_tupel_from_idx( idx_orig, tupel, iteration_bounds, 4 ); // tupel now is {ib, a, b, c}
        idx_pad = tupel[0] * spin_stride.basis + tupel[1] * spin_stride.a + tupel[2] * spin_stride.b
                  + tupel[3] * spin_stride.c;
        gradient[idx_orig][0] -= mu_s[idx_orig] * resiFFT[idx_pad] / sublattice_size;
        gradient[idx_orig][1] -= mu_s[idx_orig] * resiFFT[idx_pad + 1 * spin_stride.comp] / sublattice_size;
        gradient[idx_orig][2] -= mu_s[idx_orig] * resiFFT[idx_pad + 2 * spin_stride.comp] / sublattice_size;
    }
}
#endif

void DDI::Gradient_FFT( const vectorfield & spins, vectorfield & gradient )
{
    const auto * geometry = hamiltonian->geometry.get();

#ifdef SPIRIT_USE_CUDA
    auto & ft_D_matrices = transformed_dipole_matrices;

    auto & ft_spins = fft_plan_spins.cpx_ptr;

    auto & res_iFFT = fft_plan_reverse.real_ptr;
    auto & res_mult = fft_plan_reverse.cpx_ptr;

    int number_of_mults = it_bounds_pointwise_mult[0] * it_bounds_pointwise_mult[1] * it_bounds_pointwise_mult[2]
                          * it_bounds_pointwise_mult[3];

    FFT_Spins( spins, this->fft_plan_spins );

    // TODO: also parallelize over i_b1
    // Loop over basis atoms (i.e sublattices) and add contribution of each sublattice
    CU_FFT_Pointwise_Mult<<<( spins.size() + 1023 ) / 1024, 1024>>>(
        ft_D_matrices.data(), ft_spins.data(), res_mult.data(), it_bounds_pointwise_mult.data(),
        inter_sublattice_lookup.data(), dipole_stride, spin_stride );
    // cudaDeviceSynchronize();
    // std::cerr << "\n\n>>>>>>>>>>>  Pointwise_Mult       <<<<<<<<<\n";
    // for( int i = 0; i < 10; i++ )
    //     std::cout << ( res_mult[i].x ) << " " << ( res_mult[i].y ) << " ";
    // std::cerr << "\n>=====================================<\n\n";

    FFT::batch_iFour_3D( fft_plan_reverse );

    CU_Write_FFT_Gradients<<<( geometry->nos + 1023 ) / 1024, 1024>>>(
        res_iFFT.data(), gradient.data(), spin_stride, it_bounds_write_gradients.data(), geometry->n_cell_atoms,
        geometry->mu_s.data(), sublattice_size );
#else
    // Size of original geometry
    int Na = geometry->n_cells[0];
    int Nb = geometry->n_cells[1];
    int Nc = geometry->n_cells[2];

    FFT_Spins( spins, this->fft_plan_spins );

    auto & ft_D_matrices = transformed_dipole_matrices;
    auto & ft_spins      = fft_plan_spins.cpx_ptr;

    auto & res_iFFT = fft_plan_reverse.real_ptr;
    auto & res_mult = fft_plan_reverse.cpx_ptr;

    // Workaround for compability with intel compiler
    const int c_n_cell_atoms               = geometry->n_cell_atoms;
    const int * c_it_bounds_pointwise_mult = it_bounds_pointwise_mult.data();

// Loop over basis atoms (i.e sublattices)
#pragma omp parallel for collapse( 4 )
    for( int i_b1 = 0; i_b1 < c_n_cell_atoms; ++i_b1 )
    {
        for( int c = 0; c < c_it_bounds_pointwise_mult[2]; ++c )
        {
            for( int b = 0; b < c_it_bounds_pointwise_mult[1]; ++b )
            {
                for( int a = 0; a < c_it_bounds_pointwise_mult[0]; ++a )
                {
                    // Collect the intersublattice contributions
                    for( int i_b2 = 0; i_b2 < c_n_cell_atoms; ++i_b2 )
                    {
                        // Look up at which position the correct D-matrices are saved
                        int & b_inter = inter_sublattice_lookup[i_b1 + i_b2 * geometry->n_cell_atoms];

                        int idx_b2
                            = i_b2 * spin_stride.basis + a * spin_stride.a + b * spin_stride.b + c * spin_stride.c;
                        int idx_b1
                            = i_b1 * spin_stride.basis + a * spin_stride.a + b * spin_stride.b + c * spin_stride.c;
                        int idx_d = b_inter * dipole_stride.basis + a * dipole_stride.a + b * dipole_stride.b
                                    + c * dipole_stride.c;

                        auto & fs_x = ft_spins[idx_b2];
                        auto & fs_y = ft_spins[idx_b2 + 1 * spin_stride.comp];
                        auto & fs_z = ft_spins[idx_b2 + 2 * spin_stride.comp];

                        auto & fD_xx = ft_D_matrices[idx_d];
                        auto & fD_xy = ft_D_matrices[idx_d + 1 * dipole_stride.comp];
                        auto & fD_xz = ft_D_matrices[idx_d + 2 * dipole_stride.comp];
                        auto & fD_yy = ft_D_matrices[idx_d + 3 * dipole_stride.comp];
                        auto & fD_yz = ft_D_matrices[idx_d + 4 * dipole_stride.comp];
                        auto & fD_zz = ft_D_matrices[idx_d + 5 * dipole_stride.comp];

                        FFT::addTo(
                            res_mult[idx_b1 + 0 * spin_stride.comp],
                            FFT::mult3D( fD_xx, fD_xy, fD_xz, fs_x, fs_y, fs_z ), i_b2 == 0 );
                        FFT::addTo(
                            res_mult[idx_b1 + 1 * spin_stride.comp],
                            FFT::mult3D( fD_xy, fD_yy, fD_yz, fs_x, fs_y, fs_z ), i_b2 == 0 );
                        FFT::addTo(
                            res_mult[idx_b1 + 2 * spin_stride.comp],
                            FFT::mult3D( fD_xz, fD_yz, fD_zz, fs_x, fs_y, fs_z ), i_b2 == 0 );
                    }
                }
            } // end iteration over padded lattice cells
        }     // end iteration over second sublattice
    }

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
                for( int i_b1 = 0; i_b1 < c_n_cell_atoms; ++i_b1 )
                {
                    int idx_orig = i_b1 + geometry->n_cell_atoms * ( a + Na * ( b + Nb * c ) );
                    int idx      = i_b1 * spin_stride.basis + a * spin_stride.a + b * spin_stride.b + c * spin_stride.c;
                    gradient[idx_orig][0] -= geometry->mu_s[idx_orig] * res_iFFT[idx] / sublattice_size;
                    gradient[idx_orig][1]
                        -= geometry->mu_s[idx_orig] * res_iFFT[idx + 1 * spin_stride.comp] / sublattice_size;
                    gradient[idx_orig][2]
                        -= geometry->mu_s[idx_orig] * res_iFFT[idx + 2 * spin_stride.comp] / sublattice_size;
                }
            }
        }
    } // end iteration sublattice 1
#endif
}

void DDI::Gradient_Direct( const vectorfield & spins, vectorfield & gradient )
{
    const auto * geometry            = hamiltonian->geometry.get();
    const auto & boundary_conditions = hamiltonian->boundary_conditions;

    static constexpr scalar mult = C::mu_0 * C::mu_B * C::mu_B / ( 4 * C::Pi * 1e-30 );

    const int img_a = boundary_conditions[0] == 0 ? 0 : ddi_n_periodic_images[0];
    const int img_b = boundary_conditions[1] == 0 ? 0 : ddi_n_periodic_images[1];
    const int img_c = boundary_conditions[2] == 0 ? 0 : ddi_n_periodic_images[2];

    scalar d = 0, d3 = 0, d5 = 0;
    Vector3 diff;
    Vector3 diff_img;
    for( int idx1 = 0; idx1 < geometry->nos; idx1++ )
    {
        for( int idx2 = 0; idx2 < geometry->nos; idx2++ )
        {
            const auto & m2 = spins[idx2];

            diff       = geometry->positions[idx2] - geometry->positions[idx1];
            scalar Dxx = 0, Dxy = 0, Dxz = 0, Dyy = 0, Dyz = 0, Dzz = 0;

            for( int a_pb = -img_a; a_pb <= img_a; a_pb++ )
            {
                for( int b_pb = -img_b; b_pb <= img_b; b_pb++ )
                {
                    for( int c_pb = -img_c; c_pb <= img_c; c_pb++ )
                    {
                        diff_img
                            = diff
                              + a_pb * geometry->n_cells[0] * geometry->bravais_vectors[0] * geometry->lattice_constant
                              + b_pb * geometry->n_cells[1] * geometry->bravais_vectors[1] * geometry->lattice_constant
                              + c_pb * geometry->n_cells[2] * geometry->bravais_vectors[2] * geometry->lattice_constant;
                        d = diff_img.norm();
                        if( d > 1e-10 )
                        {
                            d3 = d * d * d;
                            d5 = d * d * d * d * d;
                            Dxx += mult * ( 3 * diff_img[0] * diff_img[0] / d5 - 1 / d3 );
                            Dxy += mult * 3 * diff_img[0] * diff_img[1] / d5; // same as Dyx
                            Dxz += mult * 3 * diff_img[0] * diff_img[2] / d5; // same as Dzx
                            Dyy += mult * ( 3 * diff_img[1] * diff_img[1] / d5 - 1 / d3 );
                            Dyz += mult * 3 * diff_img[1] * diff_img[2] / d5; // same as Dzy
                            Dzz += mult * ( 3 * diff_img[2] * diff_img[2] / d5 - 1 / d3 );
                        }
                    }
                }
            }

            gradient[idx1][0]
                -= ( Dxx * m2[0] + Dxy * m2[1] + Dxz * m2[2] ) * geometry->mu_s[idx1] * geometry->mu_s[idx2];
            gradient[idx1][1]
                -= ( Dxy * m2[0] + Dyy * m2[1] + Dyz * m2[2] ) * geometry->mu_s[idx1] * geometry->mu_s[idx2];
            gradient[idx1][2]
                -= ( Dxz * m2[0] + Dyz * m2[1] + Dzz * m2[2] ) * geometry->mu_s[idx1] * geometry->mu_s[idx2];
        }
    }
}

#ifdef SPIRIT_USE_CUDA
__global__ void CU_Write_FFT_Spin_Input(
    FFT::FFT_real_type * fft_spin_inputs, const Vector3 * spins, const int * iteration_bounds,
    FFT::StrideContainer spin_stride, const scalar * mu_s )
{
    int nos = iteration_bounds[0] * iteration_bounds[1] * iteration_bounds[2] * iteration_bounds[3];
    int tupel[4];
    int idx_pad;
    for( int idx_orig = blockIdx.x * blockDim.x + threadIdx.x; idx_orig < nos; idx_orig += blockDim.x * gridDim.x )
    {
        cu_tupel_from_idx( idx_orig, tupel, iteration_bounds, 4 ); // tupel now is {ib, a, b, c}
        idx_pad = tupel[0] * spin_stride.basis + tupel[1] * spin_stride.a + tupel[2] * spin_stride.b
                  + tupel[3] * spin_stride.c;
        fft_spin_inputs[idx_pad]                        = spins[idx_orig][0] * mu_s[idx_orig];
        fft_spin_inputs[idx_pad + 1 * spin_stride.comp] = spins[idx_orig][1] * mu_s[idx_orig];
        fft_spin_inputs[idx_pad + 2 * spin_stride.comp] = spins[idx_orig][2] * mu_s[idx_orig];
    }
}
#endif

void DDI::FFT_Spins( const vectorfield & spins, FFT::FFT_Plan & fft_plan ) const
{
    const auto * geometry = hamiltonian->geometry.get();

#ifdef SPIRIT_USE_CUDA
    CU_Write_FFT_Spin_Input<<<( geometry->nos + 1023 ) / 1024, 1024>>>(
        fft_plan.real_ptr.data(), spins.data(), it_bounds_write_spins.data(), spin_stride, geometry->mu_s.data() );
#else
    // size of original geometry
    const int Na           = geometry->n_cells[0];
    const int Nb           = geometry->n_cells[1];
    const int Nc           = geometry->n_cells[2];
    const int n_cell_atoms = geometry->n_cell_atoms;

    auto & fft_spin_inputs = fft_plan.real_ptr;

// iterate over the **original** system
#pragma omp parallel for collapse( 4 )
    for( int c = 0; c < Nc; ++c )
    {
        for( int b = 0; b < Nb; ++b )
        {
            for( int a = 0; a < Na; ++a )
            {
                for( int bi = 0; bi < n_cell_atoms; ++bi )
                {
                    int idx_orig = bi + n_cell_atoms * ( a + Na * ( b + Nb * c ) );
                    int idx      = bi * spin_stride.basis + a * spin_stride.a + b * spin_stride.b + c * spin_stride.c;

                    fft_spin_inputs[idx]                        = spins[idx_orig][0] * geometry->mu_s[idx_orig];
                    fft_spin_inputs[idx + 1 * spin_stride.comp] = spins[idx_orig][1] * geometry->mu_s[idx_orig];
                    fft_spin_inputs[idx + 2 * spin_stride.comp] = spins[idx_orig][2] * geometry->mu_s[idx_orig];
                }
            }
        }
    } // end iteration over basis
#endif
    FFT::batch_Four_3D( fft_plan );
}

#ifdef SPIRIT_USE_CUDA
__global__ void CU_Write_FFT_Dipole_Input(
    FFT::FFT_real_type * fft_dipole_inputs, const int * iteration_bounds, const Vector3 * translation_vectors,
    int n_cell_atoms, Vector3 * cell_atom_translations, const int * n_cells, int * inter_sublattice_lookup,
    const int * img, FFT::StrideContainer dipole_stride )
{
    int tupel[3];
    int sublattice_size = iteration_bounds[0] * iteration_bounds[1] * iteration_bounds[2];
    // Prefactor of ddi interaction
    static constexpr scalar mult = C::mu_0 * C::mu_B * C::mu_B / ( 4 * C::Pi * 1e-30 );
    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < sublattice_size; i += blockDim.x * gridDim.x )
    {
        cu_tupel_from_idx( i, tupel, iteration_bounds, 3 ); // tupel now is {a, b, c}
        const auto & a = tupel[0];
        const auto & b = tupel[1];
        const auto & c = tupel[2];
        int b_inter    = -1;
        for( int i_b1 = 0; i_b1 < n_cell_atoms; ++i_b1 )
        {
            for( int i_b2 = 0; i_b2 < n_cell_atoms; ++i_b2 )
            {
                if( i_b1 != i_b2 || i_b1 == 0 )
                {
                    b_inter++;
                    inter_sublattice_lookup[i_b1 + i_b2 * n_cell_atoms] = b_inter;

                    int a_idx  = a < n_cells[0] ? a : a - iteration_bounds[0];
                    int b_idx  = b < n_cells[1] ? b : b - iteration_bounds[1];
                    int c_idx  = c < n_cells[2] ? c : c - iteration_bounds[2];
                    scalar Dxx = 0, Dxy = 0, Dxz = 0, Dyy = 0, Dyz = 0, Dzz = 0;

                    Vector3 diff;

                    // Iterate over periodic images
                    for( int a_pb = -img[0]; a_pb <= img[0]; a_pb++ )
                    {
                        for( int b_pb = -img[1]; b_pb <= img[1]; b_pb++ )
                        {
                            for( int c_pb = -img[2]; c_pb <= img[2]; c_pb++ )
                            {
                                diff = ( a_idx + a_pb * n_cells[0] ) * translation_vectors[0]
                                       + ( b_idx + b_pb * n_cells[1] ) * translation_vectors[1]
                                       + ( c_idx + c_pb * n_cells[2] ) * translation_vectors[2]
                                       + cell_atom_translations[i_b1] - cell_atom_translations[i_b2];

                                if( diff.norm() > 1e-10 )
                                {
                                    auto d  = diff.norm();
                                    auto d3 = d * d * d;
                                    auto d5 = d * d * d * d * d;
                                    Dxx += mult * ( 3 * diff[0] * diff[0] / d5 - 1 / d3 );
                                    Dxy += mult * 3 * diff[0] * diff[1] / d5; // same as Dyx
                                    Dxz += mult * 3 * diff[0] * diff[2] / d5; // same as Dzx
                                    Dyy += mult * ( 3 * diff[1] * diff[1] / d5 - 1 / d3 );
                                    Dyz += mult * 3 * diff[1] * diff[2] / d5; // same as Dzy
                                    Dzz += mult * ( 3 * diff[2] * diff[2] / d5 - 1 / d3 );
                                }
                            }
                        }
                    }

                    int idx = b_inter * dipole_stride.basis + a * dipole_stride.a + b * dipole_stride.b
                              + c * dipole_stride.c;
                    fft_dipole_inputs[idx]                          = Dxx;
                    fft_dipole_inputs[idx + 1 * dipole_stride.comp] = Dxy;
                    fft_dipole_inputs[idx + 2 * dipole_stride.comp] = Dxz;
                    fft_dipole_inputs[idx + 3 * dipole_stride.comp] = Dyy;
                    fft_dipole_inputs[idx + 4 * dipole_stride.comp] = Dyz;
                    fft_dipole_inputs[idx + 5 * dipole_stride.comp] = Dzz;
                }
                else
                {
                    inter_sublattice_lookup[i_b1 + i_b2 * n_cell_atoms] = 0;
                }
            }
        }
    }
}
#endif

void DDI::FFT_Dipole_Matrices( FFT::FFT_Plan & fft_plan, const int img_a, const int img_b, const int img_c )
{
    const auto * geometry = hamiltonian->geometry.get();

#ifdef SPIRIT_USE_CUDA
    auto & fft_dipole_inputs = fft_plan.real_ptr;

    intfield img = { img_a, img_b, img_c };

    // Work around to make bravais vectors and cell_atoms available to GPU as they are currently saves as std::vectors
    // and not fields ...
    auto translation_vectors    = field<Vector3>();
    auto cell_atom_translations = field<Vector3>();

    for( int i = 0; i < 3; i++ )
        translation_vectors.push_back( geometry->lattice_constant * geometry->bravais_vectors[i] );

    for( int i = 0; i < geometry->n_cell_atoms; i++ )
        cell_atom_translations.push_back( geometry->positions[i] );

    CU_Write_FFT_Dipole_Input<<<( sublattice_size + 1023 ) / 1024, 1024>>>(
        fft_dipole_inputs.data(), it_bounds_write_dipole.data(), translation_vectors.data(), geometry->n_cell_atoms,
        cell_atom_translations.data(), geometry->n_cells.data(), inter_sublattice_lookup.data(), img.data(),
        dipole_stride );
#else
    // Prefactor of DDI
    static constexpr scalar mult = C::mu_0 * C::mu_B * C::mu_B / ( 4 * C::Pi * 1e-30 );

    // Size of original geometry
    const int Na = geometry->n_cells[0];
    const int Nb = geometry->n_cells[1];
    const int Nc = geometry->n_cells[2];

    auto & fft_dipole_inputs = fft_plan.real_ptr;

    int b_inter = -1;
    for( int i_b1 = 0; i_b1 < geometry->n_cell_atoms; ++i_b1 )
    {
        for( int i_b2 = 0; i_b2 < geometry->n_cell_atoms; ++i_b2 )
        {
            if( i_b1 == i_b2 && i_b1 != 0 )
            {
                inter_sublattice_lookup[i_b1 + i_b2 * geometry->n_cell_atoms] = 0;
                continue;
            }
            b_inter++;
            inter_sublattice_lookup[i_b1 + i_b2 * geometry->n_cell_atoms] = b_inter;

            // Iterate over the padded system
            const int * c_n_cells_padded = n_cells_padded.data();

#pragma omp parallel for collapse( 3 )
            for( int c = 0; c < c_n_cells_padded[2]; ++c )
            {
                for( int b = 0; b < c_n_cells_padded[1]; ++b )
                {
                    for( int a = 0; a < c_n_cells_padded[0]; ++a )
                    {
                        int a_idx  = a < Na ? a : a - n_cells_padded[0];
                        int b_idx  = b < Nb ? b : b - n_cells_padded[1];
                        int c_idx  = c < Nc ? c : c - n_cells_padded[2];
                        scalar Dxx = 0, Dxy = 0, Dxz = 0, Dyy = 0, Dyz = 0, Dzz = 0;
                        Vector3 diff;
                        // Iterate over periodic images
                        for( int a_pb = -img_a; a_pb <= img_a; a_pb++ )
                        {
                            for( int b_pb = -img_b; b_pb <= img_b; b_pb++ )
                            {
                                for( int c_pb = -img_c; c_pb <= img_c; c_pb++ )
                                {
                                    diff = geometry->lattice_constant
                                           * ( ( a_idx + a_pb * Na + geometry->cell_atoms[i_b1][0]
                                                 - geometry->cell_atoms[i_b2][0] )
                                                   * geometry->bravais_vectors[0]
                                               + ( b_idx + b_pb * Nb + geometry->cell_atoms[i_b1][1]
                                                   - geometry->cell_atoms[i_b2][1] )
                                                     * geometry->bravais_vectors[1]
                                               + ( c_idx + c_pb * Nc + geometry->cell_atoms[i_b1][2]
                                                   - geometry->cell_atoms[i_b2][2] )
                                                     * geometry->bravais_vectors[2] );

                                    if( diff.norm() > 1e-10 )
                                    {
                                        auto d  = diff.norm();
                                        auto d3 = d * d * d;
                                        auto d5 = d * d * d * d * d;
                                        Dxx += mult * ( 3 * diff[0] * diff[0] / d5 - 1 / d3 );
                                        Dxy += mult * 3 * diff[0] * diff[1] / d5; // same as Dyx
                                        Dxz += mult * 3 * diff[0] * diff[2] / d5; // same as Dzx
                                        Dyy += mult * ( 3 * diff[1] * diff[1] / d5 - 1 / d3 );
                                        Dyz += mult * 3 * diff[1] * diff[2] / d5; // same as Dzy
                                        Dzz += mult * ( 3 * diff[2] * diff[2] / d5 - 1 / d3 );
                                    }
                                }
                            }
                        }

                        int idx = b_inter * dipole_stride.basis + a * dipole_stride.a + b * dipole_stride.b
                                  + c * dipole_stride.c;

                        fft_dipole_inputs[idx]                          = Dxx;
                        fft_dipole_inputs[idx + 1 * dipole_stride.comp] = Dxy;
                        fft_dipole_inputs[idx + 2 * dipole_stride.comp] = Dxz;
                        fft_dipole_inputs[idx + 3 * dipole_stride.comp] = Dyy;
                        fft_dipole_inputs[idx + 4 * dipole_stride.comp] = Dyz;
                        fft_dipole_inputs[idx + 5 * dipole_stride.comp] = Dzz;
                    }
                }
            }
        }
    }
#endif
    FFT::batch_Four_3D( fft_plan );
}

void DDI::Prepare_DDI()
{
    Clean_DDI();

    if( method != DDI_Method::FFT )
        return;

    const auto * geometry            = hamiltonian->geometry.get();
    const auto & boundary_conditions = hamiltonian->boundary_conditions;

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
#if !( defined( SPIRIT_USE_CUDA ) || defined( SPIRIT_USE_FFTW ) )
    int number_of_one_dims = 0;
    for( int i = 0; i < 3; i++ )
        if( n_cells_padded[i] == 1 && ++number_of_one_dims > 1 )
            n_cells_padded[i] = 2;
#endif

    inter_sublattice_lookup.resize( geometry->n_cell_atoms * geometry->n_cell_atoms );

    // We dont need to transform over length 1 dims
    std::vector<int> fft_dims;
    for( int i = 2; i >= 0; i-- ) // Notice that reverse order is important!
    {
        if( n_cells_padded[i] > 1 )
            fft_dims.push_back( n_cells_padded[i] );
    }

    // Count how many distinct inter-lattice contributions we need to store
    // TODO: this should be expressible as a closed formula
    n_inter_sublattice = 0;
    for( int i = 0; i < geometry->n_cell_atoms; i++ )
    {
        for( int j = 0; j < geometry->n_cell_atoms; j++ )
        {
            if( i != 0 && i == j )
                continue;
            n_inter_sublattice++;
        }
    }

    // Create FFT plans
#ifdef SPIRIT_USE_CUDA
    // Set the iteration bounds for the nested for loops that are flattened in the kernels
    it_bounds_write_spins
        = { geometry->n_cell_atoms, geometry->n_cells[0], geometry->n_cells[1], geometry->n_cells[2] };

    it_bounds_write_dipole = { n_cells_padded[0], n_cells_padded[1], n_cells_padded[2] };

    it_bounds_pointwise_mult = { geometry->n_cell_atoms,
                                 ( n_cells_padded[0] / 2 + 1 ), // due to redundancy in real fft
                                 n_cells_padded[1], n_cells_padded[2] };

    it_bounds_write_gradients
        = { geometry->n_cell_atoms, geometry->n_cells[0], geometry->n_cells[1], geometry->n_cells[2] };
#endif
    auto fft_plan_dipole = FFT::FFT_Plan( fft_dims, false, 6 * n_inter_sublattice, sublattice_size );
    fft_plan_spins       = FFT::FFT_Plan( fft_dims, false, 3 * geometry->n_cell_atoms, sublattice_size );
    fft_plan_reverse     = FFT::FFT_Plan( fft_dims, true, 3 * geometry->n_cell_atoms, sublattice_size );

#if defined( SPIRIT_USE_FFTW ) || defined( SPIRIT_USE_CUDA )
    field<int *> temp_s = { &spin_stride.comp, &spin_stride.basis, &spin_stride.a, &spin_stride.b, &spin_stride.c };
    field<int *> temp_d
        = { &dipole_stride.comp, &dipole_stride.basis, &dipole_stride.a, &dipole_stride.b, &dipole_stride.c };
    ;
    FFT::get_strides( temp_s, { 3, geometry->n_cell_atoms, n_cells_padded[0], n_cells_padded[1], n_cells_padded[2] } );
    FFT::get_strides( temp_d, { 6, n_inter_sublattice, n_cells_padded[0], n_cells_padded[1], n_cells_padded[2] } );
#ifndef SPIRIT_USE_CUDA
    it_bounds_pointwise_mult = { ( n_cells_padded[0] / 2 + 1 ), // due to redundancy in real fft
                                 n_cells_padded[1], n_cells_padded[2] };
#endif
#else
    field<int *> temp_s = { &spin_stride.a, &spin_stride.b, &spin_stride.c, &spin_stride.comp, &spin_stride.basis };
    field<int *> temp_d
        = { &dipole_stride.a, &dipole_stride.b, &dipole_stride.c, &dipole_stride.comp, &dipole_stride.basis };
    ;
    FFT::get_strides( temp_s, { n_cells_padded[0], n_cells_padded[1], n_cells_padded[2], 3, geometry->n_cell_atoms } );
    FFT::get_strides( temp_d, { n_cells_padded[0], n_cells_padded[1], n_cells_padded[2], 6, n_inter_sublattice } );
    it_bounds_pointwise_mult = { n_cells_padded[0], n_cells_padded[1], n_cells_padded[2] };
    ( it_bounds_pointwise_mult[fft_dims.size() - 1] /= 2 )++;
#endif

    // Perform FFT of dipole matrices
    int img_a = boundary_conditions[0] == 0 ? 0 : ddi_n_periodic_images[0];
    int img_b = boundary_conditions[1] == 0 ? 0 : ddi_n_periodic_images[1];
    int img_c = boundary_conditions[2] == 0 ? 0 : ddi_n_periodic_images[2];

    FFT_Dipole_Matrices( fft_plan_dipole, img_a, img_b, img_c );
    transformed_dipole_matrices = std::move( fft_plan_dipole.cpx_ptr );

    if( save_dipole_matrices )
    {
        dipole_matrices = std::move( fft_plan_dipole.real_ptr );
    }
} // End prepare

void DDI::Clean_DDI()
{
    fft_plan_spins   = FFT::FFT_Plan();
    fft_plan_reverse = FFT::FFT_Plan();
}

} // namespace Interaction

} // namespace Engine
