#ifdef SPIRIT_USE_CUDA

#include "data/Geometry.hpp"
#include "engine/Vectormath_Defines.hpp"
#include <Spirit_Defines.h>
#include <engine/Hamiltonian_Heisenberg.hpp>
#include <engine/Method_MC.hpp>
#include <engine/Vectormath.hpp>
#include <iostream>
#include <fstream>
#include <utility/Constants.hpp>

using namespace Utility;

namespace Engine
{
using namespace Vectormath;

// A helper struct that contains device pointers to the relevant fields
struct Geometry_Device_Ptrs
{
    const int * n_cells;
    const scalar * mu_s;
    const int n_cell_atoms;
    const int * atom_types;

    Geometry_Device_Ptrs( const Data::Geometry & geom )
            : n_cells( geom.n_cells.data() ),
              mu_s( geom.mu_s.data() ),
              n_cell_atoms( geom.n_cell_atoms ),
              atom_types( geom.atom_types.data() )
    {
    }
};

// A helper struct that contains device pointers to the relevant fields
struct Hamiltonian_Device_Ptrs
{
    Geometry_Device_Ptrs geometry;
    const int anisotropy_n_axes;
    const scalar * anisotropy_magnitudes;
    const Vector3 * anisotropy_normals;
    const int * anisotropy_indices;

    const int exchange_n_pairs;
    const Pair * exchange_pairs;
    const scalar * exchange_magnitudes;

    const int dmi_n_pairs;
    const Pair * dmi_pairs;
    const scalar * dmi_magnitudes;
    const Vector3 * dmi_normals;

    const scalar external_field_magnitude;
    const Vector3 external_field_normal;

    const int * boundary_conditions;

    const int idx_zeeman;
    const int idx_anisotropy;
    const int idx_exchange;
    const int idx_dmi;

    Hamiltonian_Device_Ptrs( const Hamiltonian_Heisenberg & ham )
            : geometry( *ham.geometry ),
              anisotropy_n_axes( ham.anisotropy_indices.size() ),
              anisotropy_magnitudes( ham.anisotropy_magnitudes.data() ),
              anisotropy_normals( ham.anisotropy_normals.data() ),
              anisotropy_indices( ham.anisotropy_indices.data() ),
              exchange_n_pairs( ham.exchange_pairs.size() ),
              exchange_pairs( ham.exchange_pairs.data() ),
              exchange_magnitudes( ham.exchange_magnitudes.data() ),
              dmi_n_pairs( ham.dmi_pairs.size() ),
              dmi_pairs( ham.dmi_pairs.data() ),
              dmi_magnitudes( ham.dmi_magnitudes.data() ),
              dmi_normals( ham.dmi_normals.data() ),
              external_field_magnitude( ham.external_field_magnitude ),
              external_field_normal( ham.external_field_normal ),
              boundary_conditions( ham.boundary_conditions.data() ),
              idx_zeeman( ham.Idx_Zeeman() ),
              idx_anisotropy( ham.Idx_Anisotropy() ),
              idx_exchange( ham.Idx_Exchange() ),
              idx_dmi( ham.Idx_DMI() )
    {
    }
};


__device__ scalar Energy_Single_Spin( int ispin, Vector3 * spins, Hamiltonian_Device_Ptrs ham )
{
    // This function is a replacement for the Hamiltonian_Heisenberg member function of the same name, it can be used from
    // within cuda kernels
    auto & anisotropy_indices  = ham.anisotropy_indices;
    auto & anisotropy_normals  = ham.anisotropy_normals;
    auto & exchange_pairs      = ham.exchange_pairs;
    auto & exchange_magnitudes = ham.exchange_magnitudes;
    auto & dmi_pairs           = ham.dmi_pairs;
    auto & dmi_normals         = ham.dmi_normals;
    auto & dmi_magnitudes      = ham.dmi_magnitudes;
    auto & boundary_conditions = ham.boundary_conditions;
    auto & geometry            = ham.geometry;

    scalar Energy = 0;
    if( cu_check_atom_type( ham.geometry.atom_types[ispin] ) )
    {
        int icell  = ispin / ham.geometry.n_cell_atoms;
        int ibasis = ispin - icell * ham.geometry.n_cell_atoms;
        auto mu_s  = ham.geometry.mu_s;
        Pair pair_inv;

        // External field
        if( ham.idx_zeeman >= 0 )
            Energy -= mu_s[ispin] * ham.external_field_magnitude * ham.external_field_normal.dot( spins[ispin] );

        // Anisotropy
        if( ham.idx_anisotropy >= 0 )
        {
            for( int iani = 0; iani < ham.anisotropy_n_axes; ++iani )
            {
                if( anisotropy_indices[iani] == ibasis )
                {
                    if( cu_check_atom_type( ham.geometry.atom_types[ispin] ) )
                        Energy -= ham.anisotropy_magnitudes[iani]
                                  * powf( anisotropy_normals[iani].dot( spins[ispin] ), 2.0 );
                }
            }
        }

        // Exchange
        if( ham.idx_exchange >= 0 )
        {
            for( unsigned int ipair = 0; ipair < ham.exchange_n_pairs; ++ipair )
            {
                const auto & pair = exchange_pairs[ipair];
                if( pair.i == ibasis )
                {
                    int jspin = cu_idx_from_pair(
                        ispin, boundary_conditions, geometry.n_cells, geometry.n_cell_atoms, geometry.atom_types,
                        pair );
                    if( jspin >= 0 )
                        Energy -= ham.exchange_magnitudes[ipair] * spins[ispin].dot( spins[jspin] );
                }
            }
        }

        // DMI
        if( ham.idx_dmi >= 0 )
        {
            for( unsigned int ipair = 0; ipair < ham.dmi_n_pairs; ++ipair )
            {
                const auto & pair = dmi_pairs[ipair];
                if( pair.i == ibasis )
                {
                    int jspin = cu_idx_from_pair(
                        ispin, boundary_conditions, geometry.n_cells, geometry.n_cell_atoms, geometry.atom_types,
                        pair );
                    if( jspin >= 0 )
                        Energy -= ham.dmi_magnitudes[ipair]
                                  * ham.dmi_normals[ipair].dot( spins[ispin].cross( spins[jspin] ) );
                }
            }
        }

        // TODO: Quadruplets are missing, but can be added later ...
    }
    return Energy;
}

__device__ void cu_metropolis_spin_trial(
    int ispin, Vector3 * spins_old, Vector3 * spins_new, Hamiltonian_Device_Ptrs ham, const scalar rng1,
    const scalar rng2, const scalar rng3, const scalar cos_cone_angle, scalar temperature )
{

    // TODO: Implement
    // This function should perform a metropolis spin trial, using the same logic as in core/src/engine/Method_MC.cpp

    // A few things I would like to point out:
    //     1. Remember that pointers that live on the host side have *no* meaning when used on the device. This also
    //     applies to the `this` pointer of any object that was constructed on the host side. Therefore you will have to
    //     explicitly copy some parameters of the method to the device. An easy way to do this is in the argument list
    //     of the kernel invocation.
    //     2. Any type that is derived from `field` (defined in `core/include/engine/Vectormath_Defines.hpp`), uses a
    //     special allocator so that the pointers you get with the `data()` member method can be used either on the host
    //     or the device. E.g spins.data() is a Vector3 * that can be dereferenced on the host and the device.
    //     3. Only functions that are marked as __device__ can be used within kernels. That means you have to replace
    //     some functions when porting code from the cpu to the gpu.
}

__global__ void cu_parallel_metropolis( Vector3 * spins_old, Vector3 * spins_new, Hamiltonian_Device_Ptrs ham)
{
    // TODO: Implement
    // This function should perform one Iteration, meaning one spin trial for every spin in the system
    // For now I would suggest to start out with the logic of the `Parallel_Metropolis` function in `core/src/engine/Method_MC.cpp`
    // If we get that to work, we can think about further GPU specific optimization.
    // Also read my comment under the `Block_Decomposition` function in `core/src/engine/Method_MC.cpp`
}


__global__ void cu_metropolis_order( const Vector3 * spins_old, Vector3 * spins_new, int * order, unsigned int * counter, Hamiltonian_Device_Ptrs ham)
{
    // TODO: Implement such that the spins are addressed in the right order according to the block decomposition
    // The version below just accesses all spins without paying attention to the blocks

    int nos = ham.geometry.n_cells[0] * ham.geometry.n_cells[1] * ham.geometry.n_cells[2] * ham.geometry.n_cell_atoms;

    for(auto index = blockIdx.x * blockDim.x + threadIdx.x;
        index < nos;
        index +=  blockDim.x * gridDim.x)
    {
        unsigned int current_count = atomicInc(counter, nos);
        order[current_count] = index;
    }
}

void Method_MC::Parallel_Metropolis( const vectorfield & spins_old, vectorfield & spins_new )
{
    auto hamiltonian = dynamic_cast<Engine::Hamiltonian_Heisenberg *>( this->systems[0]->hamiltonian.get() );
    auto ham_ptrs = Hamiltonian_Device_Ptrs(*hamiltonian); // Collect the device pointers in a struct

    // We allocate these two fields to record tha spin-trial order of the threads
    auto order   = field<int>(spins_old.size());
    auto counter = field<unsigned int>(1, 0);

    int blockSize = 1024;
    int numBlocks = ( spins_old.size() + blockSize - 1 ) / blockSize;
    cu_metropolis_order<<<numBlocks, blockSize>>>( spins_old.data(), spins_new.data(), order.data(), counter.data(), ham_ptrs);
    cudaDeviceSynchronize();

    // dump the results in some file
    std::ofstream myfile;
    myfile.open ("mc_access_order.txt");

    myfile << "# a b c idx_spin idx_trial\n";
    // Write out the order
    auto n_cells = this->systems[0]->geometry->n_cells;

    for(int c=0; c<n_cells[2]; c++)
    {
        for(int b=0; b<n_cells[1]; b++)
        {
            for(int a=0; a<n_cells[0]; a++)
            {
                int idx = a + n_cells[0] * (b + n_cells[1] * c);
                myfile << a << " " << b << " " << c << " " << idx << " " << order[idx] << "\n";
            }
        }
    }
    myfile.close();
}

} // namespace Engine

#endif