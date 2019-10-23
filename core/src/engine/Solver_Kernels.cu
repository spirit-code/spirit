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
}
}

#endif