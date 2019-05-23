#ifdef SPIRIT_USE_CUDA

#include <engine/Solver_Kernels.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Manifoldmath.hpp>
#include <utility/Constants.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>

#include <Eigen/Dense>

#include <iostream>
#include <stdio.h>
#include <algorithm>

#include <curand.h>
#include <curand_kernel.h>

#include <cub/cub.cuh>

using namespace Utility;
using Utility::Constants::Pi;

// CUDA Version
namespace Engine
{
namespace Solver_Kernels
{
    // Utility function for the SIB Solver
    __global__
    void cu_sib_transform(const Vector3 * spins, const Vector3 * force, Vector3 * out, size_t N)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        Vector3 e1, a2, A;
        scalar detAi;
        if( idx < N )
        {
            e1 = spins[idx];
            A = 0.5 * force[idx];

            // 1/determinant(A)
            detAi = 1.0 / (1 + pow(A.norm(), 2.0));

            // calculate equation without the predictor?
            a2 = e1 - e1.cross(A);

            out[idx][0] = (a2[0] * (A[0] * A[0] + 1   ) + a2[1] * (A[0] * A[1] - A[2]) + a2[2] * (A[0] * A[2] + A[1])) * detAi;
            out[idx][1] = (a2[0] * (A[1] * A[0] + A[2]) + a2[1] * (A[1] * A[1] + 1   ) + a2[2] * (A[1] * A[2] - A[0])) * detAi;
            out[idx][2] = (a2[0] * (A[2] * A[0] - A[1]) + a2[1] * (A[2] * A[1] + A[0]) + a2[2] * (A[2] * A[2] + 1   )) * detAi;
        }
    }
    void sib_transform(const vectorfield & spins, const vectorfield & force, vectorfield & out)
    {
        int n = spins.size();
        cu_sib_transform<<<(n+1023)/1024, 1024>>>(spins.data(), force.data(), out.data(), n);
        CU_CHECK_AND_SYNC();
    }


    __global__
    void cu_line_search_a(size_t N, const Vector3 * force, const Vector3 * axis,
        const Vector3 * image_displaced, const Vector3 * force_displaced, scalar * g0, scalar * gr)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if( idx < N )
        {
            g0[idx] += force[idx].dot(axis[idx]);
            // TODO: displace dir by rotating into other spin
            // ACTUALLY: the direction is orthogonal to the rotation plane, so it does not change
            gr[idx] += ( image_displaced[idx].cross(force_displaced[idx]) ).dot(axis[idx]);
        }
    }

    __inline__ __device__
    void cu_rotate(const Vector3 & v, const Vector3 & axis, const scalar & angle, Vector3 & v_out)
    {
        v_out = v * cos(angle) + axis.cross(v) * sin(angle) +
                axis * axis.dot(v) * (1 - cos(angle));
    }

    __global__
    void cu_line_search_b(size_t N, const Vector3 * image, const Vector3 * axis, const scalar * angle,
        scalar step_size, Vector3 * image_displaced)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if( idx < N )
        {
            cu_rotate(image[idx], axis[idx], step_size * angle[idx], image_displaced[idx]);
        }
    }

    void full_inexact_line_search(const Data::Spin_System & system,
        const vectorfield & image, vectorfield & image_displaced,
        const vectorfield & force, const vectorfield & force_displaced,
        const scalarfield & angle, const vectorfield & axis, scalar & step_size, int & n_step)
    {
        // Calculate geodesic distance between image and image_displaced, if not pre-determined
        scalar r = Manifoldmath::dist_geodesic(image, image_displaced);
        if( r < 1e-6 )
        {
            step_size = 0;
            return;
        }

        scalar E0 = system.hamiltonian->Energy(image);
        // E0 = this->systems[img]->E;
        scalar Er = system.hamiltonian->Energy(image_displaced);

        size_t N = force.size();
        scalarfield field_g0(N, 0);
        scalarfield field_gr(N, 0);

        cu_line_search_a<<<(N+1023)/1024, 1024>>>(N, force.data(), axis.data(),
            image_displaced.data(), force_displaced.data(), field_g0.data(), field_gr.data());
        CU_CHECK_AND_SYNC();
        scalar g0 = Vectormath::sum(field_g0);
        scalar gr = Vectormath::sum(field_gr);

        // Approximate ine search
        ++n_step;
        step_size *= inexact_line_search(r, E0, Er, g0, gr);// * Constants::gamma / Constants::mu_B;
        cu_line_search_b<<<(N+1023)/1024, 1024>>>(N, image.data(), axis.data(), angle.data(),
            step_size, image_displaced.data());
        CU_CHECK_AND_SYNC();

        Er = system.hamiltonian->Energy(image_displaced);
        // this->Calculate_Force( this->configurations_displaced, this->forces_displaced );
        if( n_step < 20 && Er > E0+std::abs(E0)*1e-4 )
        {
            full_inexact_line_search(system, image, image_displaced, force, force_displaced, angle, axis, step_size, n_step);
        }
    }

    __global__
    void cu_ncg_beta_polak_ribiere(Vector3 * image, Vector3 * force, Vector3 * residual,
        Vector3 * residual_last, Vector3 * force_virtual, scalar * top, scalar * bot, size_t N)
    {
        static const scalar dt = 1e-3;
        // scalar dt = this->systems[0]->llg_parameters->dt;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if( idx < N )
        {
            // Set residuals
            residual_last[idx] = residual[idx];
            residual[idx] = image[idx].cross(force[idx]);
            // TODO: this is for comparison with VP etc. and needs to be fixed!
            //       in fact, all solvers should use the force, not dt*force=displacement
            force_virtual[idx] = dt * residual[idx];

            bot[idx] += residual_last[idx].dot(residual_last[idx]);
            // Polak-Ribiere formula
            // TODO: this finite difference *should* be done covariantly (i.e. displaced)
            // Vectormath::rotate(residual_last[idx], axis[idx], step_size * angle[idx], residual_last[idx]);
            top[idx] += residual[idx].dot( residual[idx] - residual_last[idx] );
            // Fletcher-Reeves formula
            // top += residual[idx].dot( residual[idx] );
        }
    }

    scalar ncg_beta_polak_ribiere(vectorfield & image, vectorfield & force, vectorfield & residual,
        vectorfield & residual_last, vectorfield & force_virtual)
    {
        size_t N = image.size();
        scalarfield field_top(N, 0);
        scalarfield field_bot(N, 0);

        cu_ncg_beta_polak_ribiere<<<(N+1023)/1024, 1024>>>(image.data(), force.data(), residual.data(),
            residual_last.data(), force_virtual.data(), field_top.data(), field_bot.data(), N);
        CU_CHECK_AND_SYNC();
        scalar top = Vectormath::sum(field_top);
        scalar bot = Vectormath::sum(field_bot);
        if( std::abs(bot) > 0 )
            return std::max(top/bot, scalar(0));
        else
            return 0;
    }

    __global__
    void cu_ncg_dir_max(Vector3 * direction, Vector3 * residual, scalar beta, Vector3 * axis, scalar * norm, size_t N)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if( idx < N )
        {
            // direction = residual + beta*direction
            direction[idx] = residual[idx] + beta*direction[idx];
            norm[idx] = direction[idx].norm();
            // direction[idx] = residual[idx] + beta[img]*residual_last[idx];
            axis[idx] = direction[idx].normalized();
            // if( dir_norm_i > dir_max )
            //     dir_max = dir_norm_i;
            // dir_avg += dir_norm_i;
            // angle[idx] = direction[idx].norm();
        }
    }

    scalar ncg_dir_max(vectorfield & direction, vectorfield & residual, scalar beta, vectorfield & axis)
    {
        size_t N = direction.size();
        scalarfield norms(N, 0);
        cu_ncg_dir_max<<<(N+1023)/1024, 1024>>>(direction.data(), residual.data(), beta, axis.data(), norms.data(), N);
        CU_CHECK_AND_SYNC();

        // Determine temporary device storage requirements
        scalarfield out(1, 0);
        void     *d_temp_storage = NULL;
        size_t   temp_storage_bytes = 0;
        cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, norms.data(), out.data(), N);
        // Allocate temporary storage
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        // Run reduction
        cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, norms.data(), out.data(), N);
        CU_CHECK_AND_SYNC();

        return out[0];
    }

    __global__
    void cu_ncg_rotate(Vector3 * direction, Vector3 * axis, scalar * angle, scalar step_size, const Vector3 * image, Vector3 * image_displaced, size_t N)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if( idx < N )
        {
            // Set rotation
            angle[idx] = step_size*direction[idx].norm();
            // Rotate
            cu_rotate(image[idx], axis[idx], angle[idx], image_displaced[idx]);
        }
    }
    void ncg_rotate(vectorfield & direction, vectorfield & axis, scalarfield & angle, scalar normalization, const vectorfield & image, vectorfield & image_displaced)
    {
        size_t N = direction.size();
        scalar step_size = 1/normalization;
        cu_ncg_rotate<<<(N+1023)/1024, 1024>>>(direction.data(), axis.data(), angle.data(), step_size, image.data(), image_displaced.data(), N);
        CU_CHECK_AND_SYNC();
    }

    __global__
    void cu_ncg_rotate_2(Vector3 * image, Vector3 * residual, Vector3 * axis, scalar * angle, scalar step_size, size_t N)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if( idx < N )
        {
            cu_rotate(image[idx], axis[idx], step_size * angle[idx], image[idx]);
            cu_rotate(residual[idx], axis[idx], step_size * angle[idx], residual[idx]);
        }
    }
    void ncg_rotate_2(vectorfield & image, vectorfield & residual, vectorfield & axis, scalarfield & angle, scalar step_size)
    {
        size_t N = image.size();
        cu_ncg_rotate_2<<<(N+1023)/1024, 1024>>>(image.data(), residual.data(), axis.data(), angle.data(), step_size, N);
        CU_CHECK_AND_SYNC();
    }
}
}

#endif