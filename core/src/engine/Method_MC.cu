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
#include <curand_kernel.h>
#include <curand.h>
#include <math.h>
#include <Eigen/Dense>
#include <Eigen/Core>

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


__device__ scalar Energy_Single_Spin( int ispin, const Vector3 * spins, Hamiltonian_Device_Ptrs ham )
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
                                  * ham.dmi_normals[ipair].dot( spins[ispin].cross( spins[jspin] ).eval() );
                    
                }
            }
        }

        // TODO: Quadruplets are missing, but can be added later ...
    }
    return Energy;
}


__device__ void cu_metropolis_spin_trial(
    int ispin, const Vector3 * spins_old, Vector3 * spins_new, Hamiltonian_Device_Ptrs ham, const scalar rng1,
    const scalar rng2, const scalar rng3, const scalar cos_cone_angle, scalar temperature, bool * kernelresult) // aditional parameter kB_T needs to be input
{

    // TODO: Implement
    // This function should perform a metropolis spin trial, using the same logic as in core/src/engine/Method_MC.cpp

    // A few things I would like to point out:
    //     1. Remember that pointers that live on the host side have no meaning when used on the device. This also
    //     applies to the `this` pointer of any object that was constructed on the host side. Therefore you will have to
    //     explicitly copy some parameters of the method to the device. An easy way to do this is in the argument list
    //     of the kernel invocation.
    //     2. Any type that is derived from `field` (defined in `core/include/engine/Vectormath_Defines.hpp`), uses a
    //     special allocator so that the pointers you get with the `data()` member method can be used either on the host
    //     or the device. E.g spins.data() is a Vector3 * that can be dereferenced on the host and the device.
    //     3. Only functions that are marked as _device_ can be used within kernels. That means you have to replace
    //     some functions when porting code from the cpu to the gpu.

    // IMP: usman: In meeting, need to ask about the data type ----
    

    Matrix3 local_basis; // usman: Ask in the meeting..., classes with functions in CUDA
    const Vector3 e_z{0,0,1};
    const scalar kB_T = Constants::k_B * temperature; //usman: kB_T Need to be explicitly inputed to the function
    // local_basis = Matrix3::Identity();
    // Calculate local basis for the spin
    
    if(fabs( spins_old[ispin].z() ) < 1 - 1e-10 ) //usman: fabs function from CUDA Math
    {
        local_basis.col( 2 ) = spins_old[ispin];
        local_basis.col( 0 ) = ( local_basis.col( 2 ).cross( e_z ).eval() ).normalized();
        local_basis.col( 1 ) = local_basis.col( 2 ).cross( local_basis.col( 0 ) ).eval();
    }
    
    else
    {
        local_basis = Matrix3::Identity();
    }
    
    // checkpoint
    //const Vector3 e_z{ 0, 0, 1 };
    // Rotation angle between 0 and cone_angle degrees
    scalar costheta = 1 - ( 1 - cos_cone_angle ) * rng1;

    //scalar sintheta = std::sqrt( 1 - costheta * costheta );
    scalar sintheta = sqrt( 1 - costheta * costheta ); //usman: sqrt function from CUDA Math

    // Random distribution of phi between 0 and 360 degrees
    scalar phi = 2 * Constants::Pi * rng2;
    
    // Vector3 local_spin_new{ sintheta * std::cos( phi ), sintheta * std::sin( phi ), costheta };
    Vector3 local_spin_new{ sintheta * cos( phi ), sintheta * sin( phi ), costheta }; //usman: sqrt function from CUDA Math
    
    // New spin orientation in regular basis
    spins_new[ispin] = local_basis * local_spin_new;

    // Energy difference of configurations with and without displacement

    // usman: !! ASK during meeting... Function "Energy_Single_Spin" not defined
    scalar Eold = Energy_Single_Spin(ispin, spins_old, ham);
    scalar Enew = Energy_Single_Spin(ispin, spins_new, ham);

    scalar Ediff = Enew - Eold;
    
    // Metropolis criterion: reject the step if energy rose
    if( Ediff > 1e-14 )
    {
        // if( this->parameters_mc->temperature < 1e-12 )
        if( temperature < 1e-12 ) // usman: Need to explicitly define the value of temperature rather than just passing the pointer
        {
            // Restore the spin
            spins_new[ispin] = spins_old[ispin];
            *kernelresult = false;
            return;
        }
        else
        {
            // Exponential factor
            
            // scalar exp_ediff = std::exp( -Ediff / kB_T ); // usman: Replaced with CUDA Exponential Function
            scalar exp_ediff = exp(-Ediff / kB_T ); // CUDA Exponential Function

            // Only reject if random number is larger than exponential
            if( exp_ediff < rng3 )
            {
                // Restore the spin
                spins_new[ispin] = spins_old[ispin];
                // Counter for the number of rejections
                *kernelresult = false;
                return;
            }
        }
    }
    
    *kernelresult = true;
    return;
    
    
}
__global__ void cu_parallel_metropolis(const Vector3 * spins_old, Vector3 * spins_new, int * order, unsigned int * counter, Hamiltonian_Device_Ptrs ham,
 int phase_a, int phase_b, int phase_c, int rest1, int rest2, int rest3, int block_size_min1, int block_size_min2, int block_size_min3, curandState *states, scalar cos_cone_angle, scalar temperature)
{
    // TODO: Implement
    // This function should perform one Iteration, meaning one spin trial for every spin in the system
    // For now I would suggest to start out with the logic of the `Parallel_Metropolis` function in `core/src/engine/Method_MC.cpp`
    // If we get that to work, we can think about further GPU specific optimization.
    // Also read my comment under the `Block_Decomposition` function in `core/src/engine/Method_MC.cpp`
    int block_size_min[3] = {block_size_min1, block_size_min2,  block_size_min3};
    int rest[3] = {rest1, rest2, rest3};
    int nos = ham.geometry.n_cells[0] * ham.geometry.n_cells[1] * ham.geometry.n_cells[2] * ham.geometry.n_cell_atoms;
    unsigned int n_blocks[3] = {gridDim.x, gridDim.y, gridDim.z};
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    int id_z = threadIdx.z + blockIdx.z * blockDim.z;
    int idx_a = 2 * id_x + phase_a;
    int idx_b = 2 * id_y + phase_b;
    int idx_c = 2 * id_z + phase_c;
    int stride_a = gridDim.x * blockDim.x;
    int stride_b = gridDim.y * blockDim.y;
    int stride_c = gridDim.z * blockDim.z;
    int seed = id_x + stride_a * id_y + stride_a * stride_b * id_z;
    curand_init(0, seed, 0, &states[seed]); //https://kth.instructure.com/courses/20917/pages/tutorial-random-numbers-in-cuda-with-curand
    bool * kernelresult;

    int i = 0;
    for(int block_c = idx_c; block_c < n_blocks[2]; block_c += stride_c * 2)
    {
        for (int block_b = idx_b; block_b < n_blocks[1]; block_b += stride_b * 2)
        {
            for (int block_a = idx_a; block_a < n_blocks[0]; block_a += stride_a * 2)
            {

                int block_size_c = (block_c == n_blocks[2] - 1) ? block_size_min[2] + rest[2] : block_size_min[2]; // Account for the remainder of division (n_cells[i] / block_size_min[i]) by increasing the block size at the edges
                int block_size_b = (block_b == n_blocks[1] - 1) ? block_size_min[1] + rest[1] : block_size_min[1];
                int block_size_a = (block_a == n_blocks[0] - 1) ? block_size_min[0] + rest[0] : block_size_min[0];
                //printf("Bl_Size_a %i, Bl_Size_b %i, Bl_Size_c %i \n", block_size_a, block_size_b, block_size_c);
                // Iterate over the current block (this has to be done serially again)
                for(int cc = 0; cc < block_size_c; cc++)
                {
                    for(int bb = 0; bb < block_size_b; bb++)
                    {
                        for(int aa = 0; aa < block_size_a; aa++)
                        {
                            for(int ibasis=0; ibasis < ham.geometry.n_cell_atoms; ibasis++)
                            {
                                int a = block_a * block_size_min[0] + aa; // We do not have to worry about the remainder of the division here, it is contained in the 'aa'/'bb'/'cc' offset
                                int b = block_b * block_size_min[1] + bb;
                                int c = block_c * block_size_min[2] + cc;

                                // Compute the current spin idx
                                int ispin = ibasis + ham.geometry.n_cell_atoms * (a + ham.geometry.n_cells[0] * (b + ham.geometry.n_cells[1] * c));
                                
                                scalar rng1 = curand_uniform(&states[seed]);
                                scalar rng2 = curand_uniform(&states[seed]);
                                scalar rng3 = curand_uniform(&states[seed]);
                                printf("seed: %i \n", seed);
                                printf("rng1: %f, rng2: %f, rng3: %f, threadx: %i, thready: %i, blockx: %i, blocky: %i \n", rng1, rng2, rng3, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
                                //printf("cone: %f, temp: %f \n", cos_cone_angle, temperature);
                                cu_metropolis_spin_trial(ispin, spins_old, spins_new, ham, rng1, rng2, rng3, cos_cone_angle, temperature, kernelresult);
                                // if (! *kernelresult) {
                                //     n_rejected ++;
                                // }
                            }
                        }
                    }
                }
            }
        }
    }
}


__global__ void cu_metropolis_order( const Vector3 * spins_old, Vector3 * spins_new, int * order, unsigned int * counter, Hamiltonian_Device_Ptrs ham,
 int phase_a, int phase_b, int phase_c, int rest1, int rest2, int rest3, int block_size_min1, int block_size_min2, int block_size_min3, curandState *states, scalar cos_cone_angle, scalar temperature)
{
   // TODO: Implement such that the spins are addressed in the right order according to the block decomposition

   
    // The version below just accesses all spins without paying attention to the blocks
    int block_size_min[3] = {block_size_min1, block_size_min2,  block_size_min3};
    int rest[3] = {rest1, rest2, rest3};
    int nos = ham.geometry.n_cells[0] * ham.geometry.n_cells[1] * ham.geometry.n_cells[2] * ham.geometry.n_cell_atoms;
    unsigned int n_blocks[3] = {gridDim.x, gridDim.y, gridDim.z};
    // for(auto index = blockIdx.x * blockDim.x + threadIdx.x;
    //     index < nos;
    //     index +=  blockDim.x * gridDim.x)
    // {
    //     unsigned int current_count = atomicInc(counter, nos);
    //     order[current_count] = index;
    // }

    // parallelized
    int idx_a = 2 * (threadIdx.x + blockIdx.x * blockDim.x) + phase_a;
    int idx_b = 2 * (threadIdx.y + blockIdx.y * blockDim.y) + phase_b;
    int idx_c = 2 * (threadIdx.z + blockIdx.z * blockDim.z) + phase_c;
    int stride_a = 2 * gridDim.x * blockDim.x;
    int stride_b = 2 * gridDim.y * blockDim.y;
    int stride_c = 2 * gridDim.z * blockDim.z;
    int seed = idx_a;
    curand_init(seed, idx_a, 0, &states[idx_a]);
    bool * kernelresult;

    int i = 0;
    for(int block_c = idx_c; block_c < n_blocks[2]; block_c += stride_c)
    {
        for (int block_b = idx_b; block_b < n_blocks[1]; block_b += stride_b)
        {
            for (int block_a = idx_a; block_a < n_blocks[0]; block_a += stride_a)
            {

                int block_size_c = (block_c == n_blocks[2] - 1) ? block_size_min[2] + rest[2] : block_size_min[2]; // Account for the remainder of division (n_cells[i] / block_size_min[i]) by increasing the block size at the edges
                int block_size_b = (block_b == n_blocks[1] - 1) ? block_size_min[1] + rest[1] : block_size_min[1];
                int block_size_a = (block_a == n_blocks[0] - 1) ? block_size_min[0] + rest[0] : block_size_min[0];
                //printf("Bl_Size_a %i, Bl_Size_b %i, Bl_Size_c %i \n", block_size_a, block_size_b, block_size_c);
                // Iterate over the current block (this has to be done serially again)
                for(int cc = 0; cc < block_size_c; cc++)
                {
                    for(int bb = 0; bb < block_size_b; bb++)
                    {
                        for(int aa = 0; aa < block_size_a; aa++)
                        {
                            for(int ibasis=0; ibasis < ham.geometry.n_cell_atoms; ibasis++)
                            {
                                int a = block_a * block_size_min[0] + aa; // We do not have to worry about the remainder of the division here, it is contained in the 'aa'/'bb'/'cc' offset
                                int b = block_b * block_size_min[1] + bb;
                                int c = block_c * block_size_min[2] + cc;

                                // Compute the current spin idx
                                int ispin = ibasis + ham.geometry.n_cell_atoms * (a + ham.geometry.n_cells[0] * (b + ham.geometry.n_cells[1] * c));
                                
                                scalar rng1 = curand_uniform(&states[idx_a]);
                                scalar rng2 = curand_uniform(&states[idx_a]);
                                scalar rng3 = curand_uniform(&states[idx_a]);
                                cu_metropolis_spin_trial(ispin, spins_old, spins_new, ham, rng1, rng2, rng3, cos_cone_angle, temperature, kernelresult);
                                // cu_metropolis_spin_trial(
                                // int ispin, Vector3 * spins_old, Vector3 * spins_new, Hamiltonian_Device_Ptrs ham, const scalar rng1,
                                // const scalar rng2, const scalar rng3, const scalar cos_cone_angle, scalar temperature, const scalar kB_T )
                                //unsigned int current_count = atomicInc(counter, nos);
                                //order[current_count] = ispin;
                                //printf("a %i, b %i, c %i, ispin %d\n", a, b, c, ispin);
                                 //printf("a_blocksize %i, b_blocksize %i, c_blocksize %i \n", block_size_a, block_size_b, block_size_c);
                                //
                                //printf("ham.geometry.n_cells[0] %i \n", ham.geometry.n_cells[0]);
                                //printf("ham.geometry.n_cells[1] %i \n", ham.geometry.n_cells[1]);
                                //printf("n_block[0]=%i, n_block[1]=%i, n_block[2]=%i \n", n_blocks[0], n_blocks[1], n_blocks[2]);
                                //printf("block_a=%i, block_b=%i, block_c=%i \n" , block_a, block_b, block_c);
                                //printf("%d \n", i);

                            }
                        }
                    }
                }
                
            }
        }
    }
}


void Method_MC::Parallel_Metropolis( const vectorfield & spins_old, vectorfield & spins_new )
{
    
    curandState *dev_random;
    cudaMalloc((void**)&dev_random, 30*30*sizeof(curandState));

    auto hamiltonian = dynamic_cast<Engine::Hamiltonian_Heisenberg *>( this->systems[0]->hamiltonian.get() );
    auto ham_ptrs = Hamiltonian_Device_Ptrs(*hamiltonian); // Collect the device pointers in a struct
    // We allocate these two fields to record tha spin-trial order of the threads
    auto order   = field<int>(spins_old.size() / 4) ;
    auto counter = field<unsigned int>(1, 0);
    //n_blocks = {4, 4, 1};
    int blockSize = 4;
    int numBlocks = 2; //( spins_old.size() + blockSize - 1 ) / blockSize;
    dim3 block(blockSize, blockSize, blockSize);
    dim3 grid(n_blocks[0], n_blocks[1], n_blocks[2]);
    auto distribution = std::uniform_real_distribution<scalar>(0, 1);
    auto temperature = this->parameters_mc->temperature;
    auto cos_cone_angle = std::cos(this->cone_angle);
    for(int phase_c = 0; phase_c < 2; phase_c++)
    {
        for(int phase_b = 0; phase_b < 2; phase_b++)
        {
            for(int phase_a = 0; phase_a < 2; phase_a++)
            {
                cu_parallel_metropolis<<<grid, block>>>( spins_old.data(), spins_new.data(), order.data(), counter.data(), ham_ptrs, 1, 0, 0, rest[0],rest[1], rest[2], 2, 2, 1, dev_random, cos_cone_angle, temperature);
            }
        }
    }
    /*
    cu_metropolis_order<<<grid, block>>>( spins_old.data(), spins_new.data(), order.data(), counter.data(), ham_ptrs, 1, 0, 0, rest[0],rest[1], rest[2], 2, 2, 1);  // cu_metropolis_order<<<grid, block>>>( spins_old.data(), spins_new.data(), order.data(), counter.data(), ham_ptrs, phase_a, phase_b, phase_c, rest[0],rest[1], rest[2], block_size_min[0], block_size_min[1], block_size_min[2]);
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
    */
}

    


} // namespace Engine

#endif

/*
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

    Matrix3 local_basis;
    const Vector3 e_z{0,0,1};
    const scalar kB_T = Constants::k_B * temperature;

    // Calculate local basis for the spin
    if( std::abs(spins_old[ispin].z()) < 1-1e-10 )
    {
        local_basis.col(2) = spins_old[ispin];
        local_basis.col(0) = (local_basis.col(2).cross(e_z)).normalized();
        local_basis.col(1) = local_basis.col(2).cross(local_basis.col(0));
    } else
    {
        local_basis = Matrix3::Identity();
    }

    // Rotation angle between 0 and cone_angle degrees
    scalar costheta = 1 - (1 - cos_cone_angle) * rng1;

    scalar sintheta = std::sqrt(1 - costheta*costheta);

    // Random distribution of phi between 0 and 360 degrees
    scalar phi = 2*Constants::Pi * rng2;

    Vector3 local_spin_new{ sintheta * std::cos(phi),
                            sintheta * std::sin(phi),
                            costheta };

    // New spin orientation in regular basis
    spins_new[ispin] = local_basis * local_spin_new;

    // Energy difference of configurations with and without displacement
    scalar Eold  = this->systems[0]->hamiltonian->Energy_Single_Spin(ispin, spins_old);
    scalar Enew  = this->systems[0]->hamiltonian->Energy_Single_Spin(ispin, spins_new);
    scalar Ediff = Enew-Eold;

    // Metropolis criterion: reject the step if energy rose
    if( Ediff > 1e-14 )
    {
        if( this->parameters_mc->temperature < 1e-12 )
        {
            // Restore the spin
            spins_new[ispin] = spins_old[ispin];
            return false;
        }
        else
        {
            // Exponential factor
            scalar exp_ediff    = std::exp( -Ediff/kB_T );
            // Only reject if random number is larger than exponential
            if( exp_ediff < rng3 )
            {
                // Restore the spin
                spins_new[ispin] = spins_old[ispin];
                // Counter for the number of rejections
                return false;
            }
        }
    }
    return true;
}
*/