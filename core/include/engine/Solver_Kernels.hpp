#pragma once
#ifndef SOLVER_KERNELS_H
#define SOLVER_KERNELS_H

#include <vector>
#include <memory>

#include <Eigen/Core>
#include <complex>
#include <engine/Vectormath_Defines.hpp>
#include <data/Spin_System.hpp>
#include <engine/Vectormath.hpp>

namespace Engine
{
namespace Solver_Kernels
{
    // SIB
    void sib_transform(const vectorfield & spins, const vectorfield & force, vectorfield & out);

    // OSO coordinates
    void oso_rotate( std::vector<std::shared_ptr<vectorfield>> & configurations, std::vector<vectorfield> & searchdir);
    void oso_calc_gradients( vectorfield & residuals, const vectorfield & spins, const vectorfield & forces);
    scalar maximum_rotation(const vectorfield & searchdir, scalar maxmove);

    // Atlas coordinates
    void atlas_calc_gradients(vector2field & residuals, const vectorfield & spins, const vectorfield & forces, const scalarfield & a3_coords);
    void atlas_rotate(std::vector<std::shared_ptr<vectorfield>> & configurations, const std::vector<scalarfield> & a3_coords, const std::vector<vector2field> & searchdir);
    bool ncg_atlas_check_coordinates(const std::vector<std::shared_ptr<vectorfield>> & spins, std::vector<scalarfield> & a3_coords, scalar tol = -0.6);
    void lbfgs_atlas_transform_direction(std::vector<std::shared_ptr<vectorfield>> & configurations, std::vector<scalarfield> & a3_coords, std::vector<std::vector<vector2field>> & atlas_updates,
                                         std::vector<std::vector<vector2field>> & grad_updates, std::vector<vector2field> & searchdir, std::vector<vector2field> & grad_pr, std::vector<scalarfield> & rho);

    // LBFGS
    template <typename Vec>
    void lbfgs_get_searchdir(int & local_iter, std::vector<scalarfield> & rho, std::vector<scalarfield> & alpha, std::vector<field<Vec>> & q_vec, std::vector<field<Vec>> & searchdir,
                             std::vector<std::vector<field<Vec>>> & delta_a, std::vector<std::vector<field<Vec>>> & delta_grad, const std::vector<field<Vec>> & grad, std::vector<field<Vec>> & grad_pr,
                             const int num_mem, const scalar maxmove) {

        static auto dot = [](const Vec & v1, const Vec &v2){return v1.dot(v2);};
        static auto set = [](const Vec & x){return x;};

        int noi = grad.size();
        int nos = grad[0].size();
        int m_index = local_iter % num_mem; // memory index
        int c_ind = 0;

        if (local_iter == 0) {  // gradient descent algorithm
            #pragma omp parallel for
            for(int img=0; img<noi; img++) {
                // Vectormath::set_c_a(1.0, grad[img], grad_pr[img]);
                Vectormath::set(grad_pr[img], grad[img], set);
                auto & dir = searchdir[img];
                auto & g_cur = grad[img];
                // Vectormath::set_c_a(-scaling, g_cur, dir);
                Vectormath::set(dir, g_cur, [](const Vec & x){return -x;});
                auto & da = delta_a[img];
                auto & dg = delta_grad[img];
                for (int i = 0; i < num_mem; i++){
                    rho[img][i] = 0.0;
                    // Vectormath::set_c_a(1.0, {0,0,0}, da[i]);
                    // Vectormath::set_c_a(1.0, {0,0,0}, dg[i]);
                    Vectormath::apply(nos, [&](int idx){da[i][idx] = Vec::Zero();});
                    Vectormath::apply(nos, [&](int idx){dg[i][idx] = Vec::Zero();});
                }
            }
        } else {
            for (int img=0; img<noi; img++) {
                // Vectormath::set_c_a(1, searchdir[img], delta_a[img][m_index]);
                Vectormath::set(delta_a[img][m_index], searchdir[img], set);

                #pragma omp parallel for
                for (int i = 0; i < nos; i++)
                    delta_grad[img][m_index][i] = grad[img][i] - grad_pr[img][i];
            }

            scalar rinv_temp = 0;
            for (int img=0; img<noi; img++) {
                // rinv_temp += Vectormath::dot(delta_grad[img][m_index], delta_a[img][m_index]);
                rinv_temp += Vectormath::reduce(delta_grad[img][m_index], delta_a[img][m_index], dot);
            }

            for (int img=0; img<noi; img++)
            {
                if (rinv_temp > 1.0e-40)
                    rho[img][m_index] = 1.0 / rinv_temp;
                else rho[img][m_index] = 1.0e40;
                if (rho[img][m_index] < 0.0)
                {
                    local_iter = 0;
                    return lbfgs_get_searchdir(local_iter, rho, alpha, q_vec, searchdir,
                            delta_a, delta_grad, grad, grad_pr, num_mem, maxmove);
                }
                // Vectormath::set_c_a(1.0, grad[img], q_vec[img]);
                Vectormath::set(q_vec[img], grad[img], set);

            }

            for (int k = num_mem - 1; k > -1; k--)
            {
                c_ind = (k + m_index + 1) % num_mem;
                scalar temp=0;
                for (int img=0; img<noi; img++)
                {
                    // temp += Vectormath::dot(delta_a[img][c_ind], q_vec[img]);
                    temp += Vectormath::reduce(delta_a[img][c_ind], q_vec[img], dot);
                }
                for (int img=0; img<noi; img++)
                {
                    alpha[img][c_ind] = rho[img][c_ind] * temp;
                    // Vectormath::add_c_a(-alpha[img][c_ind], delta_grad[img][c_ind], q_vec[img]);
                    Vectormath::apply(nos, [&](int idx){ q_vec[img][idx] += -alpha[img][c_ind] * delta_grad[img][c_ind][idx];});
                }
            }

            scalar dy2=0;
            for (int img=0; img<noi; img++)
            {
                // dy2 += Vectormath::dot(delta_grad[img][m_index], delta_grad[img][m_index]);
                dy2 += Vectormath::reduce(delta_grad[img][m_index], delta_grad[img][m_index], dot);
            }
            for (int img=0; img<noi; img++)
            {
                scalar rhody2 = dy2 * rho[img][m_index];
                scalar inv_rhody2 = 0.0;
                if (rhody2 > 1.0e-40)
                    inv_rhody2 = 1.0 / rhody2;
                else
                    inv_rhody2 = 1.0e40;
                // Vectormath::set_c_a(inv_rhody2, q_vec[img], searchdir[img]);
                Vectormath::set(searchdir[img], q_vec[img], [&](const Vec & q){return inv_rhody2 * q;} );
            }

            for (int k = 0; k < num_mem; k++)
            {
                if (local_iter < num_mem)
                        c_ind = k;
                    else
                        c_ind = (k + m_index + 1) % num_mem;

                scalar rhopdg = 0;
                for(int img=0; img<noi; img++)
                    // rhopdg += Vectormath::dot(delta_grad[img][c_ind], searchdir[img]);
                    rhopdg += Vectormath::reduce(delta_grad[img][c_ind], searchdir[img], dot);

                rhopdg *= rho[0][c_ind];

                for (int img=0; img<noi; img++)
                {
                    // Vectormath::add_c_a((alpha[img][c_ind] - rhopdg), delta_a[img][c_ind], searchdir[img]);
                    Vectormath::apply( nos,
                        [&](int idx){
                            searchdir[img][idx] += (alpha[img][c_ind] - rhopdg) * delta_a[img][c_ind][idx];
                        });
                }

            }

            for (int img=0; img<noi; img++)
            {
                // Vectormath::set_c_a(1.0, grad[img], grad_pr[img]);
                // Vectormath::set_c_a(-scaling, searchdir[img], searchdir[img]);
                Vectormath::apply(nos, [&](int idx){grad_pr[img][idx] = grad[img][idx];});
                Vectormath::apply(nos, [&](int idx){searchdir[img][idx] = -searchdir[img][idx];});

            }
        }
        local_iter++;
    }
}
}

#endif