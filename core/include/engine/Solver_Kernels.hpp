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

    // NCG
    inline scalar inexact_line_search(scalar r, scalar E0, scalar Er, scalar g0, scalar gr)
    {
        scalar c1 = -2*(Er - E0) / std::pow(r, 3) + (gr + g0) / std::pow(r, 2);
        scalar c2 = 3*(Er - E0) / std::pow(r, 2) - (gr + 2*g0) / r;
        scalar c3 = g0;
        // scalar c4 = E0;
        return std::abs( (-c2 + std::sqrt(c2*c2 - 3*c1*c3)) / (3*c1) ) / r;
    }
    scalar ncg_beta_polak_ribiere(vectorfield & image, vectorfield & force, vectorfield & residual,
        vectorfield & residual_last, vectorfield & force_virtual);
    scalar ncg_dir_max(vectorfield & direction, vectorfield & residual, scalar beta, vectorfield & axis);
    void full_inexact_line_search(const Data::Spin_System & system,
        const vectorfield & image, vectorfield & image_displaced,
        const vectorfield & force, const vectorfield & force_displaced,
        const scalarfield & angle, const vectorfield & axis, scalar & step_size, int & n_step);
    void ncg_rotate(vectorfield & direction, vectorfield & axis, scalarfield & angle,
        scalar normalization, const vectorfield & image, vectorfield & image_displaced);
    void ncg_rotate_2(vectorfield & image, vectorfield & residual, vectorfield & axis,
        scalarfield & angle, scalar step_size);



    // NCG_OSO
    void ncg_OSO_residual( vectorfield & residuals, vectorfield & residuals_last, const vectorfield & spins, const vectorfield & forces, bool approx = true);
    void ncg_OSO_a_to_spins(vectorfield & spins, const vectorfield & a_coords, const vectorfield & reference_spins);

    void ncg_OSO_update_reference_spins(vectorfield & reference_spins, vectorfield & a_coords, const vectorfield & spins);

    // void ncg_OSO_displace( std::vector<std::shared_ptr<vectorfield>> & configurations_displaced, std::vector<std::shared_ptr<vectorfield>> & reference_configurations, std::vector<vectorfield> & a_coords,
                        //    std::vector<vectorfield> & a_coords_displaced, std::vector<vectorfield> & a_directions, std::vector<bool> finish, scalarfield step_size );

    void ncg_OSO_displace( std::vector<std::shared_ptr<vectorfield>> & configurations, std::vector<vectorfield> & a_directions, scalarfield & step_size, scalar max_rot);


    void ncg_OSO_line_search( std::vector<std::shared_ptr<vectorfield>> & configurations_displaced, std::vector<vectorfield> & a_coords_displaced, std::vector<vectorfield> & a_directions,
                              std::vector<vectorfield> & forces_displaced, std::vector<vectorfield> & a_residuals_displaced, std::vector<std::shared_ptr<Data::Spin_System>> systems, std::vector<bool> & finish,
                              scalarfield & E0, scalarfield & g0, scalarfield & a_direction_norm, scalarfield & step_size  );

    inline bool ncg_OSO_wolfe_conditions(scalar E0, scalar Er, scalar g0, scalar gr, scalar step)
    {
        constexpr scalar c1 = 1e-4;
        constexpr scalar c2 = 0.9;
        return (Er <= E0 + c1 * step * gr) && (std::abs(gr) <= c2 * std::abs(g0) );
    }


    // NCG_Atlas
    void ncg_atlas_residual( vector2field & residuals, vector2field & residuals_last, const vectorfield & spins,
                             const vectorfield & forces, const scalarfield & a3_coords );
    void ncg_atlas_to_spins(const vector2field & a_coords, const scalarfield & a3_coords, vectorfield & spins);
    void ncg_spins_to_atlas(const vectorfield & spins, vector2field & a_coords, scalarfield & a3_coords);

    bool ncg_atlas_check_coordinates(const vectorfield & spins, scalarfield & a3_coords);
    void ncg_atlas_transform_direction(const vectorfield & spins, vector2field & a_coords, scalarfield & a3_coords, vector2field & a_directions);

    void ncg_atlas_displace( std::vector<std::shared_ptr<vectorfield>> & configurations_displaced, std::vector<vector2field> & a_coords, std::vector<scalarfield> & a3_coords,
                             std::vector<vector2field> & a_coords_displaced, std::vector<vector2field> & a_directions, std::vector<bool> finish, scalarfield step_size );

    void ncg_atlas_line_search(  std::vector<std::shared_ptr<vectorfield>> & configurations_displaced, std::vector<vector2field> & a_coords_displaced, std::vector<scalarfield>  & a3_coords, std::vector<vector2field> & a_directions,
                                 std::vector<vectorfield> & forces_displaced, std::vector<vector2field> & a_residuals_displaced, std::vector<std::shared_ptr<Data::Spin_System>> systems, std::vector<bool> & finish,
                                 scalarfield & E0, scalarfield & g0, scalarfield & a_direction_norm, scalarfield & step_size);


    scalar ncg_atlas_norm(const vector2field & a_coords);

    scalar ncg_atlas_distance(const vector2field & a_coords1, const vector2field & a_coords2);


    void oso_rotate( std::vector<std::shared_ptr<vectorfield>> & configurations, std::vector<vectorfield> & searchdir);

    void oso_calc_gradients( vectorfield & residuals,const vectorfield & spins, const vectorfield & forces);

    double maximum_rotation(const vectorfield & searchdir, double maxmove);

    void lbfgs_get_descent_direction(int iteration, field<int> & n_updates, field<vectorfield> & a_direction, const field<vectorfield> & residual, const field<field<vectorfield>> & a_updates, const field<field<vectorfield>> & grad_updates, const field<scalarfield> & rho_temp, field<scalarfield> & alpha_temp);
    // LBFGS_OSO
    template <typename Vec>
    void lbfgs_get_searchdir(int & local_iter, field<scalarfield> & rho, field<scalarfield> & alpha, field<field<Vec>> & q_vec, field<field<Vec>> & searchdir,
                             field<field<field<Vec>>> & delta_a, field<field<field<Vec>>> & delta_grad, const field<field<Vec>> & grad, field<field<Vec>> & grad_pr,
                             const int num_mem, const double maxmove) {

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