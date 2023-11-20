#ifdef SPIRIT_USE_CUDA

#include <engine/Indexing.hpp>
#include <engine/Vectormath_Defines.hpp>
#include <engine/interaction/Hamiltonian_Heisenberg_Kernels.cuh>
#include <utility/Constants.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

namespace C = Utility::Constants;
using Engine::Indexing::cu_check_atom_type;
using Engine::Indexing::cu_idx_from_pair;
using Engine::Indexing::cu_tupel_from_idx;

namespace Engine
{

__global__ void CU_E_Zeeman(
    const Vector3 * spins, const int * atom_types, const int n_cell_atoms, const scalar * mu_s,
    const scalar external_field_magnitude, const Vector3 external_field_normal, scalar * energy, size_t n_cells_total )
{
    for( auto icell = blockIdx.x * blockDim.x + threadIdx.x; icell < n_cells_total; icell += blockDim.x * gridDim.x )
    {
        for( int ibasis = 0; ibasis < n_cell_atoms; ++ibasis )
        {
            int ispin = n_cell_atoms * icell + ibasis;
            if( cu_check_atom_type( atom_types[ispin] ) )
                energy[ispin] -= mu_s[ispin] * external_field_magnitude * external_field_normal.dot( spins[ispin] );
        }
    }
}

__global__ void CU_E_Cubic_Anisotropy(
    const Vector3 * spins, const int * atom_types, const int n_cell_atoms, const int n_anisotropies,
    const int * anisotropy_indices, const scalar * anisotropy_magnitude, scalar * energy, size_t n_cells_total )
{
    for( auto icell = blockIdx.x * blockDim.x + threadIdx.x; icell < n_cells_total; icell += blockDim.x * gridDim.x )
    {
        for( int iani = 0; iani < n_anisotropies; ++iani )
        {
            int ispin = icell * n_cell_atoms + anisotropy_indices[iani];
            if( cu_check_atom_type( atom_types[ispin] ) )
                energy[ispin]
                    -= anisotropy_magnitude[iani] / 2
                       * ( pow( spins[ispin][0], 4.0 ) + pow( spins[ispin][1], 4.0 ) + pow( spins[ispin][2], 4.0 ) );
        }
    }
}

__global__ void CU_E_Exchange(
    const Vector3 * spins, const int * atom_types, const int * boundary_conditions, const int * n_cells,
    int n_cell_atoms, int n_pairs, const Pair * pairs, const scalar * magnitudes, scalar * energy, size_t size )
{
    int bc[3] = { boundary_conditions[0], boundary_conditions[1], boundary_conditions[2] };
    int nc[3] = { n_cells[0], n_cells[1], n_cells[2] };

    for( auto icell = blockIdx.x * blockDim.x + threadIdx.x; icell < size; icell += blockDim.x * gridDim.x )
    {
        for( auto ipair = 0; ipair < n_pairs; ++ipair )
        {
            int ispin = pairs[ipair].i + icell * n_cell_atoms;
            int jspin = cu_idx_from_pair( ispin, bc, nc, n_cell_atoms, atom_types, pairs[ipair] );
            if( jspin >= 0 )
            {
                energy[ispin] -= 0.5 * magnitudes[ipair] * spins[ispin].dot( spins[jspin] );
            }
        }
    }
}

__global__ void CU_E_DMI(
    const Vector3 * spins, const int * atom_types, const int * boundary_conditions, const int * n_cells,
    int n_cell_atoms, int n_pairs, const Pair * pairs, const scalar * magnitudes, const Vector3 * normals,
    scalar * energy, size_t size )
{
    int bc[3] = { boundary_conditions[0], boundary_conditions[1], boundary_conditions[2] };
    int nc[3] = { n_cells[0], n_cells[1], n_cells[2] };

    for( auto icell = blockIdx.x * blockDim.x + threadIdx.x; icell < size; icell += blockDim.x * gridDim.x )
    {
        for( auto ipair = 0; ipair < n_pairs; ++ipair )
        {
            int ispin = pairs[ipair].i + icell * n_cell_atoms;
            int jspin = cu_idx_from_pair( ispin, bc, nc, n_cell_atoms, atom_types, pairs[ipair] );
            if( jspin >= 0 )
            {
                energy[ispin] -= 0.5 * magnitudes[ipair] * normals[ipair].dot( spins[ispin].cross( spins[jspin] ) );
            }
        }
    }
}

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

__global__ void CU_Gradient_Cubic_Anisotropy(
    const Vector3 * spins, const int * atom_types, const int n_cell_atoms, const int n_anisotropies,
    const int * anisotropy_indices, const scalar * anisotropy_magnitude, Vector3 * gradient, size_t n_cells_total )
{
    for( auto icell = blockIdx.x * blockDim.x + threadIdx.x; icell < n_cells_total; icell += blockDim.x * gridDim.x )
    {
        for( int iani = 0; iani < n_anisotropies; ++iani )
        {
            int ispin = icell * n_cell_atoms + anisotropy_indices[iani];
            if( cu_check_atom_type( atom_types[ispin] ) )
            {
                for( int icomp = 0; icomp < 3; ++icomp )
                {
                    gradient[ispin][icomp] -= 2.0 * anisotropy_magnitude[iani] * pow( spins[ispin][icomp], 3.0 );
                }
            }
        }
    }
}

__global__ void CU_Gradient_Exchange(
    const Vector3 * spins, const int * atom_types, const int * boundary_conditions, const int * n_cells,
    int n_cell_atoms, int n_pairs, const Pair * pairs, const scalar * magnitudes, Vector3 * gradient, size_t size )
{
    int bc[3] = { boundary_conditions[0], boundary_conditions[1], boundary_conditions[2] };
    int nc[3] = { n_cells[0], n_cells[1], n_cells[2] };

    for( auto icell = blockIdx.x * blockDim.x + threadIdx.x; icell < size; icell += blockDim.x * gridDim.x )
    {
        for( auto ipair = 0; ipair < n_pairs; ++ipair )
        {
            int ispin = pairs[ipair].i + icell * n_cell_atoms;
            int jspin = cu_idx_from_pair( ispin, bc, nc, n_cell_atoms, atom_types, pairs[ipair] );
            if( jspin >= 0 )
            {
                gradient[ispin] -= magnitudes[ipair] * spins[jspin];
            }
        }
    }
}

__global__ void CU_Gradient_DMI(
    const Vector3 * spins, const int * atom_types, const int * boundary_conditions, const int * n_cells,
    int n_cell_atoms, int n_pairs, const Pair * pairs, const scalar * magnitudes, const Vector3 * normals,
    Vector3 * gradient, size_t size )
{
    int bc[3] = { boundary_conditions[0], boundary_conditions[1], boundary_conditions[2] };
    int nc[3] = { n_cells[0], n_cells[1], n_cells[2] };

    for( auto icell = blockIdx.x * blockDim.x + threadIdx.x; icell < size; icell += blockDim.x * gridDim.x )
    {
        for( auto ipair = 0; ipair < n_pairs; ++ipair )
        {
            int ispin = pairs[ipair].i + icell * n_cell_atoms;
            int jspin = cu_idx_from_pair( ispin, bc, nc, n_cell_atoms, atom_types, pairs[ipair] );
            if( jspin >= 0 )
            {
                gradient[ispin] -= magnitudes[ipair] * spins[jspin].cross( normals[ipair] );
            }
        }
    }
}

__global__ void CU_Gradient_Zeeman(
    const int * atom_types, const int n_cell_atoms, const scalar * mu_s, const scalar external_field_magnitude,
    const Vector3 external_field_normal, Vector3 * gradient, size_t n_cells_total )
{
    for( auto icell = blockIdx.x * blockDim.x + threadIdx.x; icell < n_cells_total; icell += blockDim.x * gridDim.x )
    {
        for( int ibasis = 0; ibasis < n_cell_atoms; ++ibasis )
        {
            int ispin = n_cell_atoms * icell + ibasis;
            if( cu_check_atom_type( atom_types[ispin] ) )
                gradient[ispin] -= mu_s[ispin] * external_field_magnitude * external_field_normal;
        }
    }
}

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
    FFT::FFT_real_type * resiFFT, Vector3 * gradient, FFT::StrideContainer spin_stride, int * iteration_bounds,
    int n_cell_atoms, scalar * mu_s, int sublattice_size )
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

__global__ void CU_Write_FFT_Spin_Input(
    FFT::FFT_real_type * fft_spin_inputs, const Vector3 * spins, int * iteration_bounds,
    FFT::StrideContainer spin_stride, scalar * mu_s )
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

__global__ void CU_Write_FFT_Dipole_Input(
    FFT::FFT_real_type * fft_dipole_inputs, int * iteration_bounds, const Vector3 * translation_vectors,
    int n_cell_atoms, Vector3 * cell_atom_translations, int * n_cells, int * inter_sublattice_lookup, int * img,
    FFT::StrideContainer dipole_stride )
{
    int tupel[3];
    int sublattice_size = iteration_bounds[0] * iteration_bounds[1] * iteration_bounds[2];
    // Prefactor of ddi interaction
    scalar mult = C::mu_0 * C::mu_B * C::mu_B / ( 4 * C::Pi * 1e-30 );
    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < sublattice_size; i += blockDim.x * gridDim.x )
    {
        cu_tupel_from_idx( i, tupel, iteration_bounds, 3 ); // tupel now is {a, b, c}
        auto & a    = tupel[0];
        auto & b    = tupel[1];
        auto & c    = tupel[2];
        int b_inter = -1;
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

} // namespace Engine

#endif
