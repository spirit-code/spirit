#include <engine/Backend.hpp>
#include <engine/Indexing.hpp>
#include <engine/Neighbours.hpp>
#include <engine/Vectormath.hpp>
#include <engine/spin/interaction/DDI.hpp>
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

namespace Spin
{

namespace Interaction
{

namespace
{

void Energy_per_Spin_Direct(
    const Geometry & geometry, const intfield & boundary_conditions, const DDI::Data & data, const vectorfield & spins,
    scalarfield & energy );
void Energy_per_Spin_Cutoff(
    const Geometry & geometry, const intfield & boundary_conditions, const DDI::Cache & cache,
    const vectorfield & spins, scalarfield & energy );
void Energy_per_Spin_FFT(
    const Geometry & geometry, const intfield & boundary_conditions, DDI::Cache & cache, const vectorfield & spins,
    scalarfield & energy );

void Gradient_Direct(
    const Geometry & geometry, const intfield & boundary_conditions, const DDI::Data & data, const vectorfield & spins,
    vectorfield & gradient );
void Gradient_Cutoff(
    const Geometry & geometry, const intfield & boundary_conditions, const DDI::Cache & cache,
    const vectorfield & spins, vectorfield & gradient );
void Gradient_FFT(
    const Geometry & geometry, const intfield & boundary_conditions, DDI::Cache & cache, const vectorfield & spins,
    vectorfield & gradient );

// Calculate the FT of the padded D-matrics
void FFT_Dipole_Matrices(
    const Geometry & geometry, DDI::Cache & cache, FFT::FFT_Plan & fft_plan, int img_a, int img_b, int img_c );
// Calculate the FT of the padded spins
void FFT_Spins(
    const Geometry & geometry, const vectorfield & spins, const FFT::StrideContainer & spin_stride,
    const intfield & it_bounds_write_spins, FFT::FFT_Plan & fft_plan );

// Preparations for DDI-Convolution Algorithm
void Prepare_DDI(
    const Geometry & geometry, const intfield & boundary_conditions, const DDI::Data & data, DDI::Cache & cache );
void Clean_DDI( DDI::Cache & cache );

} // namespace

void DDI::applyGeometry(
    const Geometry & geometry, const intfield & boundary_conditions, const Data & data, Cache & cache )
{
    if( data.method == DDI_Method::Cutoff )
        cache.pairs = Engine::Neighbours::Get_Pairs_in_Radius( geometry, data.cutoff_radius );
    else
        cache.pairs = field<Pair>{};

    cache.magnitudes = scalarfield( cache.pairs.size() );
    cache.normals    = vectorfield( cache.pairs.size() );

    for( std::size_t i = 0; i < cache.pairs.size(); ++i )
    {
        Engine::Neighbours::DDI_from_Pair(
            geometry,
            {
                cache.pairs[i].i,
                cache.pairs[i].j,
#ifndef SPIRIT_USE_CUDA
                cache.pairs[i].translations,
#else
                { cache.pairs[i].translations[0], cache.pairs[i].translations[1], cache.pairs[i].translations[2] }
#endif
            },
            cache.magnitudes[i], cache.normals[i] );
    };
    // Dipole-dipole
    Prepare_DDI( geometry, boundary_conditions, data, cache );

    cache.geometry            = &geometry;
    cache.boundary_conditions = &boundary_conditions;
}

template<>
void DDI::Energy::operator()( const vectorfield & spins, scalarfield & energy ) const
{
    if( !is_contributing )
        return;

    if( cache.geometry == nullptr || cache.boundary_conditions == nullptr )
        // TODO: turn this into an error
        return;

    if( data.method == DDI_Method::FFT )
        Energy_per_Spin_FFT( *cache.geometry, *cache.boundary_conditions, cache, spins, energy );
    else if( data.method == DDI_Method::Cutoff )
    {
        // TODO: Merge these implementations in the future
        if( data.cutoff_radius >= 0 )
            Energy_per_Spin_Cutoff( *cache.geometry, *cache.boundary_conditions, cache, spins, energy );
        else
            Energy_per_Spin_Direct( *cache.geometry, *cache.boundary_conditions, data, spins, energy );
    }
};

template<>
void DDI::Gradient::operator()( const vectorfield & spins, vectorfield & gradient ) const
{
    if( !is_contributing )
        return;

    if( cache.geometry == nullptr || cache.boundary_conditions == nullptr )
        // TODO: turn this into an error
        return;

    if( data.method == DDI_Method::FFT )
        Gradient_FFT( *cache.geometry, *cache.boundary_conditions, cache, spins, gradient );
    else if( data.method == DDI_Method::Cutoff )
    {
        // TODO: Merge these implementations in the future
        if( data.cutoff_radius >= 0 )
            Gradient_Cutoff( *cache.geometry, *cache.boundary_conditions, cache, spins, gradient );
        else
            Gradient_Direct( *cache.geometry, *cache.boundary_conditions, data, spins, gradient );
    }
};

// Calculate the total energy for a single spin to be used in Monte Carlo.
//      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
template<>
scalar DDI::Energy_Single_Spin::operator()( const int ispin, const vectorfield & spins ) const
{
    if( !is_contributing )
        return 0;

    // TODO
    return 0;
};

namespace
{

void Energy_per_Spin_Direct(
    const Geometry & geometry, const intfield & boundary_conditions, const DDI::Data & data, const vectorfield & spins,
    scalarfield & energy )
{
    vectorfield gradients_temp;
    gradients_temp.resize( geometry.nos );
    Vectormath::fill( gradients_temp, { 0, 0, 0 } );
    Gradient_Direct( geometry, boundary_conditions, data, spins, gradients_temp );

#pragma omp parallel for
    for( int ispin = 0; ispin < geometry.nos; ispin++ )
        energy[ispin] += 0.5 * spins[ispin].dot( gradients_temp[ispin] );
}

void Energy_per_Spin_Cutoff(
    const Geometry & geometry, const intfield & boundary_conditions, const DDI::Cache & cache,
    const vectorfield & spins, scalarfield & energy )
{
#ifdef SPIRIT_USE_CUDA
// //scalar mult = -mu_B*mu_B*1.0 / 4.0 / Pi; // multiply with mu_B^2
// scalar mult = 0.5*0.0536814951168; // mu_0*mu_B**2/(4pi*10**-30) -- the translations are in angstr�m, so the
// |r|[m] becomes |r|[m]*10^-10
// // scalar result = 0.0;

// for (unsigned int i_pair = 0; i_pair < cache.pairs.size(); ++i_pair)
// {
//     if (cache.magnitudes[i_pair] > 0.0)
//     {
//         for (int da = 0; da < geometry.n_cells[0]; ++da)
//         {
//             for (int db = 0; db < geometry.n_cells[1]; ++db)
//             {
//                 for (int dc = 0; dc < geometry.n_cells[2]; ++dc)
//                 {
//                     std::array<int, 3 > translations = { da, db, dc };
//                     // int idx_i = cache.pairs[i_pair].i;
//                     // int idx_j = cache.pairs[i_pair].j;
//                     int idx_i = idx_from_translations(geometry.n_cells, geometry.n_cell_atoms, translations);
//                     int idx_j = idx_from_translations(geometry.n_cells, geometry.n_cell_atoms, translations,
//                     cache.pairs[i_pair].translations); Energy[idx_i] -= mult / std::pow(cache.magnitudes[i_pair], 3.0) *
//                         (3 * spins[idx_j].dot(cache.normals[i_pair]) * spins[idx_i].dot(cache.normals[i_pair]) -
//                         spins[idx_i].dot(spins[idx_j]));
//                     energy[idx_j] -= mult / std::pow(cache.magnitudes[i_pair], 3.0) *
//                         (3 * spins[idx_j].dot(cache.normals[i_pair]) * spins[idx_i].dot(cache.normals[i_pair]) -
//                         spins[idx_i].dot(spins[idx_j]));
//                 }
//             }
//         }
//     }
// }
#else
    const auto & mu_s = geometry.mu_s;
    // The translations are in angstr�m, so the |r|[m] becomes |r|[m]*10^-10
    static constexpr scalar mult = C::mu_0 * C::mu_B * C::mu_B / ( 4 * C::Pi * 1e-30 );

    for( unsigned int i_pair = 0; i_pair < cache.pairs.size(); ++i_pair )
    {
        if( cache.magnitudes[i_pair] > 0.0 )
        {
            for( int da = 0; da < geometry.n_cells[0]; ++da )
            {
                for( int db = 0; db < geometry.n_cells[1]; ++db )
                {
                    for( int dc = 0; dc < geometry.n_cells[2]; ++dc )
                    {
                        std::array<int, 3> translations = { da, db, dc };

                        int ispin = cache.pairs[i_pair].i
                                    + idx_from_translations( geometry.n_cells, geometry.n_cell_atoms, translations );
                        int jspin = idx_from_pair(
                            ispin, boundary_conditions, geometry.n_cells, geometry.n_cell_atoms, geometry.atom_types,
                            cache.pairs[i_pair] );
                        if( jspin >= 0 )
                        {
                            energy[ispin] -= 0.5 * mu_s[ispin] * mu_s[jspin] * mult
                                             / std::pow( cache.magnitudes[i_pair], 3.0 )
                                             * ( 3 * spins[ispin].dot( cache.normals[i_pair] )
                                                     * spins[jspin].dot( cache.normals[i_pair] )
                                                 - spins[ispin].dot( spins[jspin] ) );
                        }
                    }
                }
            }
        }
    }
#endif
}

void Gradient_Cutoff(
    const Geometry & geometry, const intfield & boundary_conditions, const DDI::Cache & cache,
    const vectorfield & spins, vectorfield & gradient )
{
#ifdef SPIRIT_USE_CUDA
// TODO
#else
    const auto & mu_s = geometry.mu_s;
    // The translations are in angstr�m, so the |r|[m] becomes |r|[m]*10^-10
    static constexpr scalar mult = C::mu_0 * C::mu_B * C::mu_B / ( 4 * C::Pi * 1e-30 );

    for( unsigned int i_pair = 0; i_pair < cache.pairs.size(); ++i_pair )
    {
        if( cache.magnitudes[i_pair] > 0.0 )
        {
            for( int da = 0; da < geometry.n_cells[0]; ++da )
            {
                for( int db = 0; db < geometry.n_cells[1]; ++db )
                {
                    for( int dc = 0; dc < geometry.n_cells[2]; ++dc )
                    {
                        scalar skalar_contrib           = mult / std::pow( cache.magnitudes[i_pair], 3.0 );
                        std::array<int, 3> translations = { da, db, dc };

                        int ispin = cache.pairs[i_pair].i
                                    + idx_from_translations( geometry.n_cells, geometry.n_cell_atoms, translations );
                        int jspin = idx_from_pair(
                            ispin, boundary_conditions, geometry.n_cells, geometry.n_cell_atoms, geometry.atom_types,
                            cache.pairs[i_pair] );
                        if( jspin >= 0 )
                        {
                            gradient[ispin] -= mu_s[jspin] * mu_s[ispin] * skalar_contrib
                                               * ( 3 * cache.normals[i_pair] * spins[jspin].dot( cache.normals[i_pair] )
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

void Energy_per_Spin_FFT(
    const Geometry & geometry, const intfield & boundary_conditions, DDI::Cache & cache, const vectorfield & spins,
    scalarfield & energy )
{
#ifdef SPIRIT_USE_CUDA
    // TODO: maybe the gradient should be cached somehow, it is quite inefficient to calculate it
    // again just for the energy
    vectorfield gradients_temp( geometry.nos );
    Vectormath::fill( gradients_temp, { 0, 0, 0 } );
    Gradient_FFT( geometry, boundary_conditions, cache, spins, gradients_temp );
    CU_E_DDI_FFT<<<( geometry.nos + 1023 ) / 1024, 1024>>>(
        energy.data(), spins.data(), gradients_temp.data(),
        geometry.nos, geometry.n_cell_atoms, geometry.mu_s.data() );

    // === DEBUG: begin gradient comparison ===
    // vectorfield gradients_temp_dir;
    // gradients_temp_dir.resize(geometry.nos);
    // Vectormath::fill(gradients_temp_dir, {0,0,0});
    // Gradient_Direct(spins, gradients_temp_dir);

    // //get deviation
    // std::array<scalar, 3> deviation = {0,0,0};
    // std::array<scalar, 3> avg = {0,0,0};
    // std::array<scalar, 3> avg_ft = {0,0,0};

    // for(int i = 0; i < geometry.nos; i++)
    // {
    //     for(int d = 0; d < 3; d++)
    //     {
    //         deviation[d] += std::pow(gradients_temp[i][d] - gradients_temp_dir[i][d], 2);
    //         avg[d] += gradients_temp_dir[i][d];
    //         avg_ft[d] += gradients_temp[i][d];
    //     }
    // }
    // std::cerr << "Avg. Gradient (Direct) = " << avg[0]/geometry.nos << " " << avg[1]/geometry.nos << "
    // " << avg[2]/geometry.nos << std::endl; std::cerr << "Avg. Gradient (FFT)    = " <<
    // avg_ft[0]/geometry.nos << " " << avg_ft[1]/geometry.nos << " " << avg_ft[2]/geometry.nos <<
    // std::endl; std::cerr << "Relative Error in %    = " << (avg_ft[0]/avg[0]-1)*100 << " " <<
    // (avg_ft[1]/avg[1]-1)*100 << " " << (avg_ft[2]/avg[2]-1)*100 << std::endl; std::cerr << "Avg. Deviation = " <<
    // std::pow(deviation[0]/geometry.nos, 0.5) << " " << std::pow(deviation[1]/geometry.nos, 0.5) << " " <<
    // std::pow(deviation[2]/geometry.nos, 0.5) << std::endl; std::cerr << " ---------------- " << std::endl;
    // ==== DEBUG: end gradient comparison ====

#else
    // scalar Energy_DDI = 0;
    vectorfield gradients_temp;
    gradients_temp.resize( geometry.nos );
    Vectormath::fill( gradients_temp, { 0, 0, 0 } );
    Gradient_FFT( geometry, boundary_conditions, cache, spins, gradients_temp );

    // TODO: add dot_scaled to Vectormath and use that
#pragma omp parallel for
    for( int ispin = 0; ispin < geometry.nos; ispin++ )
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

void Gradient_FFT(
    const Geometry & geometry, const intfield &, DDI::Cache & cache, const vectorfield & spins, vectorfield & gradient )
{
#ifdef SPIRIT_USE_CUDA
    auto & ft_D_matrices = cache.transformed_dipole_matrices;

    auto & ft_spins = cache.fft_plan_spins.cpx_ptr;

    auto & res_iFFT = cache.fft_plan_reverse.real_ptr;
    auto & res_mult = cache.fft_plan_reverse.cpx_ptr;

    int number_of_mults = cache.it_bounds_pointwise_mult[0] * cache.it_bounds_pointwise_mult[1]
                          * cache.it_bounds_pointwise_mult[2] * cache.it_bounds_pointwise_mult[3];

    FFT_Spins( geometry, spins, cache.spin_stride, cache.it_bounds_write_spins, cache.fft_plan_spins );

    // TODO: also parallelize over i_b1
    // Loop over basis atoms (i.e sublattices) and add contribution of each sublattice
    CU_FFT_Pointwise_Mult<<<( spins.size() + 1023 ) / 1024, 1024>>>(
        ft_D_matrices.data(), ft_spins.data(),
        res_mult.data(), cache.it_bounds_pointwise_mult.data(),
        cache.inter_sublattice_lookup.data(), cache.dipole_stride, cache.spin_stride );
    // cudaDeviceSynchronize();
    // std::cerr << "\n\n>>>>>>>>>>>  Pointwise_Mult       <<<<<<<<<\n";
    // for( int i = 0; i < 10; i++ )
    //     std::cout << ( res_mult[i].x ) << " " << ( res_mult[i].y ) << " ";
    // std::cerr << "\n>=====================================<\n\n";

    FFT::batch_iFour_3D( cache.fft_plan_reverse );

    CU_Write_FFT_Gradients<<<( geometry.nos + 1023 ) / 1024, 1024>>>(
        res_iFFT.data(), gradient.data(), cache.spin_stride,
        cache.it_bounds_write_gradients.data(), geometry.n_cell_atoms,
        geometry.mu_s.data(), cache.sublattice_size );
#else
    // Size of original geometry
    int Na = geometry.n_cells[0];
    int Nb = geometry.n_cells[1];
    int Nc = geometry.n_cells[2];

    FFT_Spins( geometry, spins, cache.spin_stride, cache.it_bounds_write_spins, cache.fft_plan_spins );

    auto & ft_D_matrices = cache.transformed_dipole_matrices;
    auto & ft_spins      = cache.fft_plan_spins.cpx_ptr;

    auto & res_iFFT = cache.fft_plan_reverse.real_ptr;
    auto & res_mult = cache.fft_plan_reverse.cpx_ptr;

    // Workaround for compability with intel compiler
    const int c_n_cell_atoms               = geometry.n_cell_atoms;
    const int * c_it_bounds_pointwise_mult = cache.it_bounds_pointwise_mult.data();

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
                        const int & b_inter = cache.inter_sublattice_lookup[i_b1 + i_b2 * geometry.n_cell_atoms];

                        const int idx_b2 = i_b2 * cache.spin_stride.basis + a * cache.spin_stride.a
                                           + b * cache.spin_stride.b + c * cache.spin_stride.c;
                        const int idx_b1 = i_b1 * cache.spin_stride.basis + a * cache.spin_stride.a
                                           + b * cache.spin_stride.b + c * cache.spin_stride.c;
                        const int idx_d = b_inter * cache.dipole_stride.basis + a * cache.dipole_stride.a
                                          + b * cache.dipole_stride.b + c * cache.dipole_stride.c;

                        auto & fs_x = ft_spins[idx_b2];
                        auto & fs_y = ft_spins[idx_b2 + 1 * cache.spin_stride.comp];
                        auto & fs_z = ft_spins[idx_b2 + 2 * cache.spin_stride.comp];

                        auto & fD_xx = ft_D_matrices[idx_d];
                        auto & fD_xy = ft_D_matrices[idx_d + 1 * cache.dipole_stride.comp];
                        auto & fD_xz = ft_D_matrices[idx_d + 2 * cache.dipole_stride.comp];
                        auto & fD_yy = ft_D_matrices[idx_d + 3 * cache.dipole_stride.comp];
                        auto & fD_yz = ft_D_matrices[idx_d + 4 * cache.dipole_stride.comp];
                        auto & fD_zz = ft_D_matrices[idx_d + 5 * cache.dipole_stride.comp];

                        FFT::addTo(
                            res_mult[idx_b1 + 0 * cache.spin_stride.comp],
                            FFT::mult3D( fD_xx, fD_xy, fD_xz, fs_x, fs_y, fs_z ), i_b2 == 0 );
                        FFT::addTo(
                            res_mult[idx_b1 + 1 * cache.spin_stride.comp],
                            FFT::mult3D( fD_xy, fD_yy, fD_yz, fs_x, fs_y, fs_z ), i_b2 == 0 );
                        FFT::addTo(
                            res_mult[idx_b1 + 2 * cache.spin_stride.comp],
                            FFT::mult3D( fD_xz, fD_yz, fD_zz, fs_x, fs_y, fs_z ), i_b2 == 0 );
                    }
                }
            } // end iteration over padded lattice cells
        }     // end iteration over second sublattice
    }

    // Inverse Fourier Transform
    FFT::batch_iFour_3D( cache.fft_plan_reverse );

    // Workaround for compability with intel compiler
    const int * c_n_cells = geometry.n_cells.data();

    // Place the gradients at the correct positions and mult with correct mu
    for( int c = 0; c < c_n_cells[2]; ++c )
    {
        for( int b = 0; b < c_n_cells[1]; ++b )
        {
            for( int a = 0; a < c_n_cells[0]; ++a )
            {
                for( int i_b1 = 0; i_b1 < c_n_cell_atoms; ++i_b1 )
                {
                    const int idx_orig = i_b1 + geometry.n_cell_atoms * ( a + Na * ( b + Nb * c ) );
                    const int idx = i_b1 * cache.spin_stride.basis + a * cache.spin_stride.a + b * cache.spin_stride.b
                                    + c * cache.spin_stride.c;
                    gradient[idx_orig][0] -= geometry.mu_s[idx_orig] * res_iFFT[idx] / cache.sublattice_size;
                    gradient[idx_orig][1]
                        -= geometry.mu_s[idx_orig] * res_iFFT[idx + 1 * cache.spin_stride.comp] / cache.sublattice_size;
                    gradient[idx_orig][2]
                        -= geometry.mu_s[idx_orig] * res_iFFT[idx + 2 * cache.spin_stride.comp] / cache.sublattice_size;
                }
            }
        }
    } // end iteration sublattice 1
#endif
}

void Gradient_Direct(
    const Geometry & geometry, const intfield & boundary_conditions, const DDI::Data & data, const vectorfield & spins,
    vectorfield & gradient )
{
    static constexpr scalar mult = C::mu_0 * C::mu_B * C::mu_B / ( 4 * C::Pi * 1e-30 );

    const int img_a = boundary_conditions[0] == 0 ? 0 : data.n_periodic_images[0];
    const int img_b = boundary_conditions[1] == 0 ? 0 : data.n_periodic_images[1];
    const int img_c = boundary_conditions[2] == 0 ? 0 : data.n_periodic_images[2];

    scalar d = 0, d3 = 0, d5 = 0;
    Vector3 diff;
    Vector3 diff_img;
    for( int idx1 = 0; idx1 < geometry.nos; idx1++ )
    {
        for( int idx2 = 0; idx2 < geometry.nos; idx2++ )
        {
            const auto & m2 = spins[idx2];

            diff       = geometry.positions[idx2] - geometry.positions[idx1];
            scalar Dxx = 0, Dxy = 0, Dxz = 0, Dyy = 0, Dyz = 0, Dzz = 0;

            for( int a_pb = -img_a; a_pb <= img_a; a_pb++ )
            {
                for( int b_pb = -img_b; b_pb <= img_b; b_pb++ )
                {
                    for( int c_pb = -img_c; c_pb <= img_c; c_pb++ )
                    {
                        diff_img
                            = diff
                              + a_pb * geometry.n_cells[0] * geometry.bravais_vectors[0] * geometry.lattice_constant
                              + b_pb * geometry.n_cells[1] * geometry.bravais_vectors[1] * geometry.lattice_constant
                              + c_pb * geometry.n_cells[2] * geometry.bravais_vectors[2] * geometry.lattice_constant;
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
                -= ( Dxx * m2[0] + Dxy * m2[1] + Dxz * m2[2] ) * geometry.mu_s[idx1] * geometry.mu_s[idx2];
            gradient[idx1][1]
                -= ( Dxy * m2[0] + Dyy * m2[1] + Dyz * m2[2] ) * geometry.mu_s[idx1] * geometry.mu_s[idx2];
            gradient[idx1][2]
                -= ( Dxz * m2[0] + Dyz * m2[1] + Dzz * m2[2] ) * geometry.mu_s[idx1] * geometry.mu_s[idx2];
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

void FFT_Spins(
    const Geometry & geometry, const vectorfield & spins, const FFT::StrideContainer & spin_stride,
    const intfield & it_bounds_write_spins, FFT::FFT_Plan & fft_plan )
{
#ifdef SPIRIT_USE_CUDA
    CU_Write_FFT_Spin_Input<<<( geometry.nos + 1023 ) / 1024, 1024>>>(
        fft_plan.real_ptr.data(), spins.data(),
        it_bounds_write_spins.data(), spin_stride, geometry.mu_s.data() );
#else
    // size of original geometry
    const int Na           = geometry.n_cells[0];
    const int Nb           = geometry.n_cells[1];
    const int Nc           = geometry.n_cells[2];
    const int n_cell_atoms = geometry.n_cell_atoms;

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

                    fft_spin_inputs[idx]                        = spins[idx_orig][0] * geometry.mu_s[idx_orig];
                    fft_spin_inputs[idx + 1 * spin_stride.comp] = spins[idx_orig][1] * geometry.mu_s[idx_orig];
                    fft_spin_inputs[idx + 2 * spin_stride.comp] = spins[idx_orig][2] * geometry.mu_s[idx_orig];
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
    const int n_cell_atoms, Vector3 * cell_atom_translations, const int * n_cells, int * inter_sublattice_lookup,
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

void FFT_Dipole_Matrices(
    const Geometry & geometry, DDI::Cache & cache, FFT::FFT_Plan & fft_plan, const int img_a, const int img_b,
    const int img_c )
{
#ifdef SPIRIT_USE_CUDA
    auto & fft_dipole_inputs = fft_plan.real_ptr;

    intfield img = { img_a, img_b, img_c };

    // Work around to make bravais vectors and cell_atoms available to GPU as they are currently saves as std::vectors
    // and not fields ...
    auto translation_vectors    = field<Vector3>();
    auto cell_atom_translations = field<Vector3>();

    for( int i = 0; i < 3; i++ )
        translation_vectors.push_back( geometry.lattice_constant * geometry.bravais_vectors[i] );

    for( int i = 0; i < geometry.n_cell_atoms; i++ )
        cell_atom_translations.push_back( geometry.positions[i] );

    static constexpr int blockSize = 768;
    CU_Write_FFT_Dipole_Input<<<( cache.sublattice_size + blockSize - 1 ) / blockSize, blockSize>>>(
        fft_dipole_inputs.data(), cache.it_bounds_write_dipole.data(),
        translation_vectors.data(), geometry.n_cell_atoms,
        cell_atom_translations.data(), geometry.n_cells.data(),
        cache.inter_sublattice_lookup.data(), img.data(), cache.dipole_stride );
#else
    // Prefactor of DDI
    static constexpr scalar mult = C::mu_0 * C::mu_B * C::mu_B / ( 4 * C::Pi * 1e-30 );

    // Size of original geometry
    const int Na = geometry.n_cells[0];
    const int Nb = geometry.n_cells[1];
    const int Nc = geometry.n_cells[2];

    auto & fft_dipole_inputs = fft_plan.real_ptr;

    int b_inter = -1;
    for( int i_b1 = 0; i_b1 < geometry.n_cell_atoms; ++i_b1 )
    {
        for( int i_b2 = 0; i_b2 < geometry.n_cell_atoms; ++i_b2 )
        {
            if( i_b1 == i_b2 && i_b1 != 0 )
            {
                cache.inter_sublattice_lookup[i_b1 + i_b2 * geometry.n_cell_atoms] = 0;
                continue;
            }
            b_inter++;
            cache.inter_sublattice_lookup[i_b1 + i_b2 * geometry.n_cell_atoms] = b_inter;

            // Iterate over the padded system
            const int * c_n_cells_padded = cache.n_cells_padded.data();

#pragma omp parallel for collapse( 3 )
            for( int c = 0; c < c_n_cells_padded[2]; ++c )
            {
                for( int b = 0; b < c_n_cells_padded[1]; ++b )
                {
                    for( int a = 0; a < c_n_cells_padded[0]; ++a )
                    {
                        const int a_idx = a < Na ? a : a - cache.n_cells_padded[0];
                        const int b_idx = b < Nb ? b : b - cache.n_cells_padded[1];
                        const int c_idx = c < Nc ? c : c - cache.n_cells_padded[2];
                        scalar Dxx = 0, Dxy = 0, Dxz = 0, Dyy = 0, Dyz = 0, Dzz = 0;
                        // Iterate over periodic images
                        for( int a_pb = -img_a; a_pb <= img_a; a_pb++ )
                        {
                            for( int b_pb = -img_b; b_pb <= img_b; b_pb++ )
                            {
                                for( int c_pb = -img_c; c_pb <= img_c; c_pb++ )
                                {
                                    const Vector3 diff = geometry.lattice_constant
                                                         * ( ( a_idx + a_pb * Na + geometry.cell_atoms[i_b1][0]
                                                               - geometry.cell_atoms[i_b2][0] )
                                                                 * geometry.bravais_vectors[0]
                                                             + ( b_idx + b_pb * Nb + geometry.cell_atoms[i_b1][1]
                                                                 - geometry.cell_atoms[i_b2][1] )
                                                                   * geometry.bravais_vectors[1]
                                                             + ( c_idx + c_pb * Nc + geometry.cell_atoms[i_b1][2]
                                                                 - geometry.cell_atoms[i_b2][2] )
                                                                   * geometry.bravais_vectors[2] );

                                    if( diff.norm() > 1e-10 )
                                    {
                                        const auto d  = diff.norm();
                                        const auto d3 = d * d * d;
                                        const auto d5 = d * d * d * d * d;
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

                        const int idx = b_inter * cache.dipole_stride.basis + a * cache.dipole_stride.a
                                        + b * cache.dipole_stride.b + c * cache.dipole_stride.c;

                        fft_dipole_inputs[idx]                                = Dxx;
                        fft_dipole_inputs[idx + 1 * cache.dipole_stride.comp] = Dxy;
                        fft_dipole_inputs[idx + 2 * cache.dipole_stride.comp] = Dxz;
                        fft_dipole_inputs[idx + 3 * cache.dipole_stride.comp] = Dyy;
                        fft_dipole_inputs[idx + 4 * cache.dipole_stride.comp] = Dyz;
                        fft_dipole_inputs[idx + 5 * cache.dipole_stride.comp] = Dzz;
                    }
                }
            }
        }
    }
#endif
    FFT::batch_Four_3D( fft_plan );
}

void Prepare_DDI(
    const Geometry & geometry, const intfield & boundary_conditions, const DDI::Data & data, DDI::Cache & cache )
{
    Clean_DDI( cache );

    if( data.method != DDI_Method::FFT )
        return;

    // We perform zero-padding in a lattice direction if the dimension of the system is greater than 1 *and*
    //  - the boundary conditions are open, or
    //  - the boundary conditions are periodic and zero-padding is explicitly requested
    cache.n_cells_padded.resize( 3 );
    for( int i = 0; i < 3; i++ )
    {
        cache.n_cells_padded[i]   = geometry.n_cells[i];
        bool perform_zero_padding = geometry.n_cells[i] > 1 && ( boundary_conditions[i] == 0 || data.pb_zero_padding );
        if( perform_zero_padding )
            cache.n_cells_padded[i] *= 2;
    }
    cache.sublattice_size = cache.n_cells_padded[0] * cache.n_cells_padded[1] * cache.n_cells_padded[2];

    FFT::FFT_Init();

// Workaround for bug in kissfft
// kissfft_ndr does not perform one-dimensional FFTs properly
#if !( defined( SPIRIT_USE_CUDA ) || defined( SPIRIT_USE_FFTW ) )
    int number_of_one_dims = 0;
    for( int i = 0; i < 3; i++ )
        if( cache.n_cells_padded[i] == 1 && ++number_of_one_dims > 1 )
            cache.n_cells_padded[i] = 2;
#endif

    cache.inter_sublattice_lookup.resize( geometry.n_cell_atoms * geometry.n_cell_atoms );

    // We dont need to transform over length 1 dims
    std::vector<int> fft_dims;
    for( int i = 2; i >= 0; i-- ) // Notice that reverse order is important!
    {
        if( cache.n_cells_padded[i] > 1 )
            fft_dims.push_back( cache.n_cells_padded[i] );
    }

    // Count how many distinct inter-lattice contributions we need to store
    // TODO: this should be expressible as a closed formula
    cache.n_inter_sublattice = 0;
    for( int i = 0; i < geometry.n_cell_atoms; i++ )
    {
        for( int j = 0; j < geometry.n_cell_atoms; j++ )
        {
            if( i != 0 && i == j )
                continue;
            cache.n_inter_sublattice++;
        }
    }

    // Create FFT plans
#ifdef SPIRIT_USE_CUDA
    // Set the iteration bounds for the nested for loops that are flattened in the kernels
    cache.it_bounds_write_spins
        = { geometry.n_cell_atoms, geometry.n_cells[0], geometry.n_cells[1], geometry.n_cells[2] };

    cache.it_bounds_write_dipole = { cache.n_cells_padded[0], cache.n_cells_padded[1], cache.n_cells_padded[2] };

    cache.it_bounds_pointwise_mult = { geometry.n_cell_atoms,
                                       ( cache.n_cells_padded[0] / 2 + 1 ), // due to redundancy in real fft
                                       cache.n_cells_padded[1], cache.n_cells_padded[2] };

    cache.it_bounds_write_gradients
        = { geometry.n_cell_atoms, geometry.n_cells[0], geometry.n_cells[1], geometry.n_cells[2] };
#endif
    auto fft_plan_dipole   = FFT::FFT_Plan( fft_dims, false, 6 * cache.n_inter_sublattice, cache.sublattice_size );
    cache.fft_plan_spins   = FFT::FFT_Plan( fft_dims, false, 3 * geometry.n_cell_atoms, cache.sublattice_size );
    cache.fft_plan_reverse = FFT::FFT_Plan( fft_dims, true, 3 * geometry.n_cell_atoms, cache.sublattice_size );

#if defined( SPIRIT_USE_FFTW ) || defined( SPIRIT_USE_CUDA )
    field<int *> temp_s = { &cache.spin_stride.comp, &cache.spin_stride.basis, &cache.spin_stride.a,
                            &cache.spin_stride.b, &cache.spin_stride.c };
    field<int *> temp_d = { &cache.dipole_stride.comp, &cache.dipole_stride.basis, &cache.dipole_stride.a,
                            &cache.dipole_stride.b, &cache.dipole_stride.c };
    ;
    FFT::get_strides(
        temp_s,
        { 3, geometry.n_cell_atoms, cache.n_cells_padded[0], cache.n_cells_padded[1], cache.n_cells_padded[2] } );
    FFT::get_strides(
        temp_d,
        { 6, cache.n_inter_sublattice, cache.n_cells_padded[0], cache.n_cells_padded[1], cache.n_cells_padded[2] } );
#ifndef SPIRIT_USE_CUDA
    cache.it_bounds_pointwise_mult = { ( cache.n_cells_padded[0] / 2 + 1 ), // due to redundancy in real fft
                                       cache.n_cells_padded[1], cache.n_cells_padded[2] };
#endif
#else
    field<int *> temp_s = { &cache.spin_stride.a, &cache.spin_stride.b, &cache.spin_stride.c, &cache.spin_stride.comp,
                            &cache.spin_stride.basis };
    field<int *> temp_d = { &cache.dipole_stride.a, &cache.dipole_stride.b, &cache.dipole_stride.c,
                            &cache.dipole_stride.comp, &cache.dipole_stride.basis };
    ;
    FFT::get_strides(
        temp_s,
        { cache.n_cells_padded[0], cache.n_cells_padded[1], cache.n_cells_padded[2], 3, geometry.n_cell_atoms } );
    FFT::get_strides(
        temp_d,
        { cache.n_cells_padded[0], cache.n_cells_padded[1], cache.n_cells_padded[2], 6, cache.n_inter_sublattice } );

    cache.it_bounds_pointwise_mult = { cache.n_cells_padded[0], cache.n_cells_padded[1], cache.n_cells_padded[2] };
    ( cache.it_bounds_pointwise_mult[fft_dims.size() - 1] /= 2 )++;
#endif

    // Perform FFT of dipole matrices
    const int img_a = boundary_conditions[0] == 0 ? 0 : data.n_periodic_images[0];
    const int img_b = boundary_conditions[1] == 0 ? 0 : data.n_periodic_images[1];
    const int img_c = boundary_conditions[2] == 0 ? 0 : data.n_periodic_images[2];

    FFT_Dipole_Matrices( geometry, cache, fft_plan_dipole, img_a, img_b, img_c );
    cache.transformed_dipole_matrices = std::move( fft_plan_dipole.cpx_ptr );

    if( cache.save_dipole_matrices )
    {
        cache.dipole_matrices = std::move( fft_plan_dipole.real_ptr );
    }
} // End prepare

void Clean_DDI( DDI::Cache & cache )
{
    cache.fft_plan_spins   = FFT::FFT_Plan();
    cache.fft_plan_reverse = FFT::FFT_Plan();
}

} // namespace

} // namespace Interaction

} // namespace Spin

} // namespace Engine
