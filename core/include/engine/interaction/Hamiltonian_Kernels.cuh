#pragma once
#ifndef SPIRIT_CORE_ENGINE_HAMILTONIAN_HEISENBERG_KERNELS_CUH
#define SPIRIT_CORE_ENGINE_HAMILTONIAN_HEISENBERG_KERNELS_CUH

#ifdef SPIRIT_USE_CUDA

#include <Spirit/Hamiltonian.h>
#include <Spirit/Spirit_Defines.h>
#include <data/Geometry.hpp>
#include <engine/FFT.hpp>
#include <engine/Vectormath_Defines.hpp>

namespace Engine
{

__global__ void CU_E_Exchange(
    const Vector3 * spins, const int * atom_types, const int * boundary_conditions, const int * n_cells,
    int n_cell_atoms, int n_pairs, const Pair * pairs, const scalar * magnitudes, scalar * energy, size_t size );

__global__ void CU_E_DMI(
    const Vector3 * spins, const int * atom_types, const int * boundary_conditions, const int * n_cells,
    int n_cell_atoms, int n_pairs, const Pair * pairs, const scalar * magnitudes, const Vector3 * normals,
    scalar * energy, size_t size );

// TODO: add dot_scaled to Vectormath and use that
__global__ void CU_E_DDI_FFT(
    scalar * energy, const Vector3 * spins, const Vector3 * gradients, const int nos, const int n_cell_atoms,
    const scalar * mu_s );

__global__ void CU_Gradient_Exchange(
    const Vector3 * spins, const int * atom_types, const int * boundary_conditions, const int * n_cells,
    int n_cell_atoms, int n_pairs, const Pair * pairs, const scalar * magnitudes, Vector3 * gradient, size_t size );

__global__ void CU_Gradient_DMI(
    const Vector3 * spins, const int * atom_types, const int * boundary_conditions, const int * n_cells,
    int n_cell_atoms, int n_pairs, const Pair * pairs, const scalar * magnitudes, const Vector3 * normals,
    Vector3 * gradient, size_t size );

__global__ void CU_FFT_Pointwise_Mult(
    FFT::FFT_cpx_type * ft_D_matrices, FFT::FFT_cpx_type * ft_spins, FFT::FFT_cpx_type * res_mult,
    int * iteration_bounds, int * inter_sublattice_lookup, FFT::StrideContainer dipole_stride,
    FFT::StrideContainer spin_stride );

__global__ void CU_Write_FFT_Gradients(
    FFT::FFT_real_type * resiFFT, Vector3 * gradient, FFT::StrideContainer spin_stride, int * iteration_bounds,
    int n_cell_atoms, scalar * mu_s, int sublattice_size );

__global__ void CU_Write_FFT_Spin_Input(
    FFT::FFT_real_type * fft_spin_inputs, const Vector3 * spins, int * iteration_bounds,
    FFT::StrideContainer spin_stride, scalar * mu_s );

__global__ void CU_Write_FFT_Dipole_Input(
    FFT::FFT_real_type * fft_dipole_inputs, int * iteration_bounds, const Vector3 * translation_vectors,
    int n_cell_atoms, Vector3 * cell_atom_translations, int * n_cells, int * inter_sublattice_lookup, int * img,
    FFT::StrideContainer dipole_stride );

} // namespace Engine

#endif

#endif
