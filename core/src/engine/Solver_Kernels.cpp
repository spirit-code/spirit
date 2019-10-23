#include <engine/Solver_Kernels.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Manifoldmath.hpp>
#include <utility/Constants.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>

#include <Eigen/Dense>

#include <array>
#include <algorithm>

using namespace Utility;
using Utility::Constants::Pi;

namespace Engine
{
namespace Solver_Kernels
{
    #ifndef SPIRIT_USE_CUDA

    void sib_transform(const vectorfield & spins, const vectorfield & force, vectorfield & out)
    {
        #pragma omp parallel for
        for (unsigned int i = 0; i < spins.size(); ++i)
        {
            Vector3 A = 0.5 * force[i];

            // 1/determinant(A)
            scalar detAi = 1.0 / (1 + A.squaredNorm());

            // calculate equation without the predictor?
            Vector3 a2 = spins[i] - spins[i].cross(A);

            out[i][0] = (a2[0] * (A[0] * A[0] + 1   ) + a2[1] * (A[0] * A[1] - A[2]) + a2[2] * (A[0] * A[2] + A[1])) * detAi;
            out[i][1] = (a2[0] * (A[1] * A[0] + A[2]) + a2[1] * (A[1] * A[1] + 1   ) + a2[2] * (A[1] * A[2] - A[0])) * detAi;
            out[i][2] = (a2[0] * (A[2] * A[0] - A[1]) + a2[1] * (A[2] * A[1] + A[0]) + a2[2] * (A[2] * A[2] + 1   )) * detAi;
        }
    }

    void oso_calc_gradients(vectorfield & grad,  const vectorfield & spins, const vectorfield & forces)
    {
        #pragma omp parallel for
        for( int i=0; i<spins.size(); i++)
        {
            Vector3 temp = -spins[i].cross(forces[i]);
            grad[i][0] =  temp[2];
            grad[i][1] = -temp[1];
            grad[i][2] =  temp[0];
        }
    }

    void oso_rotate( std::vector<std::shared_ptr<vectorfield>> & configurations, std::vector<vectorfield> & searchdir)
    {
        int noi = configurations.size();
        int nos = configurations[0]->size();
        for(int img=0; img<noi; ++img)
        {
            Matrix3 tmp;
            Matrix3 A_prime;
            for( int i=0; i<nos; i++)
            {
                scalar theta = (searchdir[img][i]).norm();
                if(theta < 1.0e-20)
                {
                    tmp = Matrix3::Identity();
                } else {
                    A_prime <<                         0,  -searchdir[img][i][0], -searchdir[img][i][1],
                            searchdir[img][i][0],                        0, -searchdir[img][i][2],
                            searchdir[img][i][1], searchdir[img][i][2],                       0;

                    A_prime /= theta;
                    tmp = Matrix3::Identity() + sin(theta) * A_prime + (1-cos(theta)) * A_prime * A_prime;
                }
                (*configurations[img])[i] = tmp * (*configurations[img])[i] ;
            }
        }
    }

    scalar maximum_rotation(const vectorfield & searchdir, scalar maxmove){
        int nos = searchdir.size();
        scalar theta_rms = 0;
        #pragma omp parallel for reduction(+:theta_rms)
        for(int i=0; i<nos; ++i)
            theta_rms += (searchdir[i]).squaredNorm();
        theta_rms = sqrt(theta_rms/nos);
        scalar scaling = (theta_rms > maxmove) ? maxmove/theta_rms : 1.0;
        return scaling;
    }

    void atlas_rotate(std::vector<std::shared_ptr<vectorfield>> & configurations, const field<scalarfield> & a3_coords, const std::vector<vector2field> & searchdir)
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

    void lbfgs_atlas_transform_direction(field<std::shared_ptr<vectorfield>> & configurations, field<scalarfield> & a3_coords, field<field<vector2field>> & atlas_updates, field<field<vector2field>> & grad_updates, field<vector2field> & searchdir, field<vector2field> & grad_pr, field<scalarfield> & rho)
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

    #endif
}
}