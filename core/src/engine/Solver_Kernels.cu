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

#include <engine/Backend_par.hpp>

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

    void oso_calc_gradients(vectorfield & grad, const vectorfield & spins, const vectorfield & forces)
    {
        const Matrix3 t = ( Matrix3() << 0,0,1,0,-1,0,1,0,0 ).finished();

        auto g=grad.data();
        auto s=spins.data();
        auto f=forces.data();

        Backend::par::apply( spins.size(), [g,s,f,t] SPIRIT_LAMBDA (int idx)
            {
                g[idx] = t * (-s[idx].cross(f[idx]));
            }
        );
    }

    void oso_rotate( std::vector<std::shared_ptr<vectorfield>> & configurations, std::vector<vectorfield> & searchdir)
    {
        int noi = configurations.size();
        int nos = configurations[0]->size();
        for(int img=0; img<noi; ++img)
        {

            auto s  = configurations[img]->data();
            auto sd = searchdir[img].data();
 
            // tmp <<  q+z*z*w, s1+p1, s2+p2,
            //         s1-p1, q+y*y*w, s3+p3,
            //         s2-p2, s3-p3, q+x*x*w;
            
            Backend::par::apply( nos, [s, sd] SPIRIT_LAMBDA (int idx) 
                {
                    scalar theta = (sd[idx]).norm();
                    scalar q = cos(theta), w = 1-q, 
                           x = -sd[idx][0]/theta, y = -sd[idx][1]/theta, z = -sd[idx][2]/theta,
                           s1 = -y*z*w, s2 = x*z*w, s3 = -x*y*w,
                           p1 = x*sin(theta), p2 = y*sin(theta), p3 = z*sin(theta);

                    scalar t1, t2, t3;
                    if(theta > 1.0e-20) // if theta is too small we do nothing
                    {
                        t1 = (q+z*z*w) * s[idx][0] + (s1+p1)   * s[idx][1] + (s2+p2)   * s[idx][2];
                        t2 = (s1-p1)   * s[idx][0] + (q+y*y*w) * s[idx][1] + (s3+p3)   * s[idx][2];
                        t3 = (s2-p2)   * s[idx][0] + (s3-p3)   * s[idx][1] + (q+x*x*w) * s[idx][2];
                        s[idx][0] = t1;
                        s[idx][1] = t2;
                        s[idx][2] = t3;
                    };
                }
            );
        }
    }

    scalar maximum_rotation(const vectorfield & searchdir, scalar maxmove)
    {
        int nos = searchdir.size();
        scalar theta_rms = 0;
        theta_rms = sqrt( Backend::par::reduce(searchdir, [] SPIRIT_LAMBDA (const Vector3 & v){ return v.squaredNorm(); }) / nos );
        scalar scaling = (theta_rms > maxmove) ? maxmove/theta_rms : 1.0;
        return scaling;
    }

    void atlas_rotate(std::vector<std::shared_ptr<vectorfield>> & configurations, const std::vector<scalarfield> & a3_coords, const std::vector<vector2field> & searchdir)
    {
        int noi = configurations.size();
        int nos = configurations[0]->size();
        for(int img=0; img<noi; img++ )
        {
            field<Vector3> & spins = *configurations[img];
            const field<Vector2> & d = searchdir[img];
            #pragma omp parallel for
            for(int i=0; i < nos; i++)
            {
                const scalar gamma = (1 + spins[i][2] * a3_coords[img][i]);
                const scalar denom = (spins[i].head<2>().squaredNorm())/gamma + 2 * d[i].head<2>().dot( spins[i].head<2>() ) + gamma * d[i].head<2>().squaredNorm();
                spins[i].head<2>() = 2*(spins[i].head<2>() + d[i]*gamma);
                spins[i][2] = a3_coords[img][i] * (gamma - denom);
                spins[i] *= 1/(gamma + denom);
            }
        }
    }

    void atlas_calc_gradients(vector2field & residuals, const vectorfield & spins, const vectorfield & forces, const scalarfield & a3_coords)
    {
        Eigen::Matrix<scalar, 3,2 > J;
        #pragma omp parallel for
        for(int i=0; i < spins.size(); i++)
        {
            const auto & s  = spins[i];
            const auto & a3 = a3_coords[i];

            J(0,0) =  s[1]*s[1] + s[2]*(s[2] + a3);
            J(0,1) = -s[0]*s[1];
            J(1,0) = -s[0]*s[1];
            J(1,1) =  s[0]*s[0]  + s[2]*(s[2] + a3);
            J(2,0) = -s[0]*(s[2] + a3);
            J(2,1) = -s[1]*(s[2] + a3);
            residuals[i] = -forces[i].transpose() * J;
        }
    }

    bool ncg_atlas_check_coordinates(const std::vector<std::shared_ptr<vectorfield>> & spins, std::vector<scalarfield> & a3_coords, scalar tol)
    {
        int noi = spins.size();
        int nos = (*spins[0]).size();
        // Check if we need to reset the maps
        bool result = false;
        for(int img=0; img<noi; img++)
        {
            #pragma omp parallel for
            for( int i=0; i<nos; i++ )
            {
                // If for one spin the z component deviates too much from the pole we perform a reset for *all* spins
                // Note: I am not sure why we reset for all spins ... but this is in agreement with the method of F. Rybakov
                // printf("blabla %f\n", (*spins[img])[i][2]*a3_coords[img][i] );
                if( (*spins[img])[i][2]*a3_coords[img][i] < tol )
                {
                    result = true;
                }
            }
        }
        return result;
    }

    void lbfgs_atlas_transform_direction(std::vector<std::shared_ptr<vectorfield>> & configurations, std::vector<scalarfield> & a3_coords, std::vector<field<vector2field>> & atlas_updates, std::vector<field<vector2field>> & grad_updates, std::vector<vector2field> & searchdir, std::vector<vector2field> & grad_pr, std::vector<scalarfield> & rho)
    {
        int noi = configurations.size();
        int nos = configurations[0]->size();

        for(int img=0; img<noi; img++)
        {
            for(int n=0; n<atlas_updates[img].size(); n++)
            {
                rho[img][n] = 1/rho[img][n];
            }
        }

        for(int img=0; img<noi; img++)
        {
            scalar factor = 1;
            #pragma omp parallel for
            for( int i=0; i<nos; ++i )
            {
                const auto & s =  (*configurations[img])[i];
                auto &      a3 = a3_coords[img][i];

                if( s[2]*a3 < 0 )
                {
                    // Transform coordinates to optimal map
                    a3 = (s[2] > 0) ? 1 : -1;
                    factor = (1 - a3 * s[2]) / (1 + a3 * s[2]);
                    searchdir[img][i]  *= factor;
                    grad_pr[img][i]    *= factor;

                    for(int n=0; n<atlas_updates[img].size(); n++)
                    {
                        rho[img][n] += (factor-1)*(factor-1) * atlas_updates[img][n][i].dot(grad_updates[img][n][i]);
                        atlas_updates[img][n][i] *= factor;
                        grad_updates[img][n][i]  *= factor;
                    }
                }
            }
        }

        for(int img=0; img<noi; img++)
        {
            for(int n=0; n<atlas_updates[img].size(); n++)
            {
                rho[img][n] = 1/rho[img][n];
            }
        }
    }
}
}

#endif