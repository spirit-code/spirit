#ifdef USE_CUDA

#include <Vectormath.hpp>
#include <Manifoldmath.hpp>

#include <cmath>
#include <iostream>
#include <stdio.h>

// CUDA Version
namespace Engine
{
	namespace Vectormath
	{
        scalar norm(const vectorfield & vf)
        {
            scalar x = dot(vf, vf);
            cudaDeviceSynchronize();
            return std::sqrt(x);
        }

        void normalize(vectorfield & vf)
        {
            scalar sc = 1.0/norm(vf);
            scale(vf, sc);
            cudaDeviceSynchronize();
        }


        void project_parallel(vectorfield & vf1, const vectorfield & vf2)
        {
            vectorfield vf3 = vf1;
            project_orthogonal(vf3, vf2);
            add_c_a(-1, vf3, vf1);
            cudaDeviceSynchronize();
        }

        __global__ void cu_project_orthogonal(Vector3 *vf1, const Vector3 *vf2, scalar proj, size_t N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            if(idx < N)
            {
                vf1[idx] -= proj*vf2[idx];
            }
        }

        // The wrapper for the calling of the actual kernel
        void project_orthogonal(vectorfield & vf1, const vectorfield & vf2)
        {        
            int n = vf1.size();

            // Get projection
            scalar proj=dot(vf1, vf2);
            // Project vf1
            cu_project_orthogonal<<<(n+1023)/1024, 1024>>>(vf1.data(), vf2.data(), proj, n);
            cudaDeviceSynchronize();
        }

        void invert_parallel(vectorfield & vf1, const vectorfield & vf2)
        {
            scalar proj=dot(vf1, vf2);
            add_c_a(-2*proj, vf2, vf1);
            cudaDeviceSynchronize();
        }
        
        void invert_orthogonal(vectorfield & vf1, const vectorfield & vf2)
        {
            vectorfield vf3 = vf1;
            project_orthogonal(vf3, vf2);
            add_c_a(-2, vf3, vf1);
            cudaDeviceSynchronize();
        }
    }
}

#endif