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

        scalar g0 = 0;
        scalar gr = 0;
        #pragma omp parallel for reduction(+:g0) reduction(+:gr)
        for( int i=0; i<image.size(); ++i )
        {
            g0 += force[i].dot(axis[i]);
            // TODO: displace dir by rotating into other spin
            // ACTUALLY: the direction is orthogonal to the rotation plane, so it does not change
            gr += ( image_displaced[i].cross(force_displaced[i]) ).dot(axis[i]);
        }

        // Approximate ine search
        ++n_step;
        step_size *= inexact_line_search(r, E0, Er, g0, gr);// * Constants::gamma / Constants::mu_B;
        #pragma omp parallel for
        for( int i=0; i<image.size(); ++i )
        {
            Vectormath::rotate(image[i], axis[i], step_size * angle[i], image_displaced[i]);
        }
        Er = system.hamiltonian->Energy(image_displaced);
        // this->Calculate_Force( this->configurations_displaced, this->forces_displaced );
        if( n_step < 20 && Er > E0+std::abs(E0)*1e-4 )
        {
            full_inexact_line_search(system, image, image_displaced, force, force_displaced, angle, axis, step_size, n_step);
        }
    }

    scalar ncg_beta_polak_ribiere(vectorfield & image, vectorfield & force, vectorfield & residual,
        vectorfield & residual_last, vectorfield & force_virtual)
    {
        scalar dt = 1e-3;
        // scalar dt = this->systems[0]->llg_parameters->dt;

        scalar top=0, bot=0;

        #pragma omp parallel for
        for( int i=0; i<image.size(); ++i )
        {
            // Set residuals
            residual_last[i] = residual[i];
            residual[i] = image[i].cross(force[i]);
            // TODO: this is for comparison with VP etc. and needs to be fixed!
            //       in fact, all solvers should use the force, not dt*force=displacement
            force_virtual[i] = dt * residual[i];

            bot += residual_last[i].dot(residual_last[i]);
            // Polak-Ribiere formula
            // TODO: this finite difference *should* be done covariantly (i.e. displaced)
            // Vectormath::rotate(residual_last[i], axis[i], step_size * angle[i], residual_last[i]);
            top += residual[i].dot( residual[i] - residual_last[i] );
            // Fletcher-Reeves formula
            // top += residual[i].dot( residual[i] );
        }
        if( std::abs(bot) > 0 )
            return std::max(top/bot, scalar(0));
        else
            return 0;
    }

    scalar ncg_dir_max(vectorfield & direction, vectorfield & residual, scalar beta, vectorfield & axis)
    {
        scalar dir_max = 0;
        #pragma omp parallel for reduction(max : dir_max)
        for( int i=0; i<direction.size(); ++i )
        {
            // direction = residual + beta*direction
            direction[i] = residual[i] + beta*direction[i];
            scalar dir_norm_i = direction[i].norm();
            // direction[i] = residual[i] + beta[img]*residual_last[i];
            axis[i] = direction[i].normalized();
            if( dir_norm_i > dir_max )
                dir_max = dir_norm_i;
            // dir_avg += dir_norm_i;
            // angle[i] = direction[i].norm();
        }
        return dir_max;
    }

    void ncg_rotate(vectorfield & direction, vectorfield & axis, scalarfield & angle, scalar normalization, const vectorfield & image, vectorfield & image_displaced)
    {
        #pragma omp parallel for
        for( int i=0; i<image.size(); ++i )
        {
            // Set rotation
            angle[i] = direction[i].norm() / normalization;
            // Rotate
            Vectormath::rotate(image[i], axis[i], angle[i], image_displaced[i]);
        }
    }

    void ncg_rotate_2(vectorfield & image, vectorfield & residual, vectorfield & axis, scalarfield & angle, scalar step_size)
    {
        #pragma omp parallel for
        for( int i=0; i<image.size(); ++i )
        {
            Vectormath::rotate(image[i], axis[i], step_size * angle[i], image[i]);
            Vectormath::rotate(residual[i], axis[i], step_size * angle[i], residual[i]);
        }
    }

    // NCG_OSO
    // Calculates the residuals for a certain spin configuration
    void ncg_OSO_residual( vectorfield & a_residuals, vectorfield & a_residuals_last, const vectorfield & spins, const vectorfield & forces, bool approx)
    {
        using cpx = std::complex<scalar>;

        if(approx) // Use approximate gradient
        {
            #pragma omp parallel for
            for( int i=0; i<spins.size(); i++)
            {
                a_residuals_last[i] = a_residuals[i];
                Vector3 temp = -spins[i].cross(forces[i]);
                a_residuals[i][0] =  temp[2];
                a_residuals[i][1] = -temp[1];
                a_residuals[i][2] =  temp[0];
            }
        } else {
        }
    }

    // Transforms the a coordinates to spins
    void ncg_OSO_a_to_spins(vectorfield & spins, const vectorfield & a_coords, const vectorfield & reference_configuration)
    {
        using cpx = std::complex<scalar>;
        #pragma omp parallel for
        for( int i =0; i<spins.size(); i++ )
        {
            const scalar & a = a_coords[i][0], b = a_coords[i][1], c = a_coords[i][2];
            scalar x    = sqrt(a*a + b*b + c*c);
            scalar pref = 1/( x * sqrt(2 * (a*a + c*c)) );
            cpx lambda2 = cpx(0,-x);
            cpx lambda3 = cpx(0,x);
            Vector3c L1 = {1/x * c, 1/x * -b, 1/x * a};
            Vector3c L2 = {cpx(b*c, a*x), a*a + c*c, cpx(a*b, -c*x) };
            Vector3c L  = {1, exp(lambda2), exp(lambda3)};
            Eigen::Matrix<cpx, 3, 3> V;
            V << L1, pref*L2, (pref*L2).conjugate();
            spins[i] = (V * L.asDiagonal() * V.adjoint() * reference_configuration[i]).real();
        }
    }

    void ncg_OSO_update_reference_spins(vectorfield & reference_spins, vectorfield & a_coords, const vectorfield & spins)
    {
        #pragma omp parallel for
        for( int i=0; i<spins.size(); ++i )
        {
            a_coords[i] = {0,0,0};
            reference_spins[i] = spins[i];
        }
    }

    void ncg_OSO_displace( std::vector<std::shared_ptr<vectorfield>> & configurations, std::vector<vectorfield> & a_directions, scalarfield & step_size, scalar max_rot)
    {
        int noi = configurations.size();
        int nos = configurations[0]->size();
        for(int img=0; img<noi; ++img)
        {
            scalar theta_rms = 0;
            #pragma omp parallel for reduction(+:theta_rms)
            for(int i=0; i<nos; ++i)
                theta_rms += (a_directions[img][i]).squaredNorm();
            theta_rms = sqrt(theta_rms)/nos;

            auto scaling = (theta_rms > max_rot) ? max_rot/theta_rms : 1.0;
            step_size[img] *= scaling;

            // Debug
            // if(scaling < 1.0)
            // {
            //     std::cout << "WARNING applying scaling with " << scaling <<"\n";
            //     std::cout << "theta_rms/Pi is " << theta_rms/Constants::Pi << "\n";
            // }

            Matrix3 tmp;
            Matrix3 A_prime;

            for( int i=0; i<nos; i++)
            {
                scalar theta = (a_directions[img][i]).norm();

                if(theta < 1e-20)
                {
                    tmp = Matrix3::Identity();
                } else {

                    // Ugly version
                    // scalar  q = cos(theta), w = 1-q,
                    //         x = a_directions[img][i][0]/theta, y = a_directions[img][i][1]/theta, z = a_directions[img][i][2]/theta,
                    //         s1 = -y*z*w, s2 = x*z*w, s3 = -x*y*w,
                    //         p1 = x * sin(theta), p2 = y * sin(theta), p3 = z * sin(theta);

                    // tmp <<  q+z*z*w, s1+p1, s2+p2,
                    //         s1-p1, q+y*y*w, s3+p3,
                    //         s2-p2, s3-p3, q+x*x*w;

                    A_prime <<                         0,  a_directions[img][i][0], a_directions[img][i][1],
                                -a_directions[img][i][0],                        0, a_directions[img][i][2],
                                -a_directions[img][i][1], -a_directions[img][i][2],                       0;

                    A_prime /= theta;
                    tmp = Matrix3::Identity() + sin(theta*scaling) * A_prime + (1-cos(theta*scaling)) * A_prime * A_prime;
                }
                (*configurations[img])[i] = tmp * (*configurations[img])[i] ;
            }
        }
    }

    void ncg_OSO_line_search( std::vector<std::shared_ptr<vectorfield>> & configurations_displaced, std::vector<vectorfield> & a_coords_displaced, std::vector<vectorfield> & a_directions,
                              std::vector<vectorfield> & forces_displaced, std::vector<vectorfield> & a_residuals_displaced, std::vector<std::shared_ptr<Data::Spin_System>> systems, std::vector<bool> & finish,
                              scalarfield & E0, scalarfield & g0, scalarfield & a_direction_norm, scalarfield & step_size )
    {
        int noi = configurations_displaced.size();
        int nos = configurations_displaced[0]->size();

        for( int img=0; img<noi; ++img )
        {
            if(finish[img])
                continue;

            // Calculate displaced energy
            scalar Er = systems[img]->hamiltonian->Energy(*configurations_displaced[img]);

            if( std::abs(Er - E0[img]) < 1e-16 )
            {
                fmt::print("[ Line search ] Energy too close\n");
                // step_size[img] = 0;
                finish[img] = true;
                continue;
            }

            // Calculate displaced residual
            Solver_Kernels::ncg_OSO_residual(a_residuals_displaced[img], a_residuals_displaced[img], *configurations_displaced[img], forces_displaced[img]);

            // Calculate displaced directional derivative
            scalar gr = 0;
            #pragma omp parallel for reduction(+:gr)
            for( int i=0; i<nos; ++i )
            {
                gr -= a_residuals_displaced[img][i].dot(a_directions[img][i])/a_direction_norm[img];
            }

            fmt::print("[ Line search ] E0   = {}\n", E0[img]);
            fmt::print("[ Line search ] Er   = {}\n", Er);
            fmt::print("[ Line search ] diff = {}\n", Er-E0[img]);
            fmt::print("[ Line search ] g0   = {}\n", g0[img]);
            fmt::print("[ Line search ] gr   = {}\n", gr);

            // If wolfe conditions are fulfilled terminate, else suggest new step size
            if( ncg_OSO_wolfe_conditions(E0[img], Er, g0[img], gr, step_size[img]*a_direction_norm[img]) )
            {
                fmt::print("[ Line search ] >>> Finished <<<\n");
                finish[img] = true;
            } else {
                scalar factor = inexact_line_search(step_size[img]*a_direction_norm[img], E0[img], Er, g0[img], gr);
                fmt::print("[ Line search ] Continueing ...\n");
                fmt::print("[ Line search ] factor = {}\n", factor);
                if(!std::isnan(factor))
                {
                    step_size[img] *= factor;
                } else {
                    step_size[img] *= 0.5;
                }
            }
        }
    }


    // NCG Atlas
    void ncg_atlas_residual( vector2field & residuals, vector2field & residuals_last, const vectorfield & spins,
                             const vectorfield & forces, const scalarfield & a3_coords )
    {
        #pragma omp parallel for
        for(int i=0; i < spins.size(); i++)
        {
            Eigen::Matrix<scalar, 3,2 > J;
            const auto & s  = spins[i];
            const auto & a3 = a3_coords[i];

            J(0,0) =  s[1]*s[1] + s[2]*(s[2] + a3);
            J(0,1) = -s[0]*s[1];
            J(1,0) = -s[0]*s[1];
            J(1,1) =  s[0]*s[0]  + s[2]*(s[2] + a3);
            J(2,0) = -s[0]*(s[2] + a3);
            J(2,1) = -s[1]*(s[2] + a3);

            // If the two adresses are different we save the old residuals
            if( &residuals != &residuals_last )
                residuals_last[i] = residuals[i];

            residuals[i] = forces[i].transpose() * J;
        }
    }

    void ncg_atlas_to_spins(const vector2field & atlas_coords, const scalarfield & a3_coords, vectorfield & spins)
    {
        #pragma omp_parallel_for
        for(int i=0; i<atlas_coords.size(); i++)
        {
            auto &        s = spins[i];
            const auto &  a = atlas_coords[i];
            const auto & a3 = a3_coords[i];

            s[0] = 2*a[0] / (1 + a[0]*a[0] + a[1]*a[1]);
            s[1] = 2*a[1] / (1 + a[0]*a[0] + a[1]*a[1]);
            s[2] = a3 * (1 - a[0]*a[0] - a[1]*a[1]) / (1 + a[0]*a[0] + a[1]*a[1]);
        }
    }

    void ncg_spins_to_atlas(const vectorfield & spins, vector2field & atlas_coords, scalarfield & a3_coords)
    {
        #pragma omp_parallel_for
        for(int i=0; i<spins.size(); i++)
        {
            const auto & s = spins[i];
            auto &       a = atlas_coords[i];
            auto &      a3 = a3_coords[i];

            a3 = (s[2] > 0) ? 1 : -1;
            a[0] = s[0] / (1 + s[2]*a3);
            a[1] = s[1] / (1 + s[2]*a3);
        }
    }

    bool ncg_atlas_check_coordinates(const field<std::shared_ptr<vectorfield>> & spins, field<scalarfield> & a3_coords, scalar tol)
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

    void ncg_atlas_transform_direction(const vectorfield & spins, vector2field & a_coords, scalarfield & a3_coords, vector2field & a_directions)
    {
        #pragma omp parallel for
        for( int i=0; i<spins.size(); ++i )
        {
            const auto & s = spins[i];
            auto &       a = a_coords[i];
            auto &      a3 = a3_coords[i];

            if( spins[i][2]*a3_coords[i] < 0 )
            {
                // Transform coordinates to optimal map
                a3 = (s[2] > 0) ? 1 : -1;
                a[0] = s[0] / (1 + s[2]*a3);
                a[1] = s[1] / (1 + s[2]*a3);

                // Also transform search direction to new map
                a_directions[i] *= (1 - a3 * s[2]) / (1 + a3 * s[2]);
            }
        }
    }

    void ncg_atlas_displace( std::vector<std::shared_ptr<vectorfield>> & configurations_displaced, std::vector<vector2field> & a_coords, std::vector<scalarfield> & a3_coords,
                             std::vector<vector2field> & a_coords_displaced, std::vector<vector2field> & a_directions, std::vector<bool> finish, scalarfield step_size )
    {
        int noi = configurations_displaced.size();
        int nos = configurations_displaced[0]->size();

        for(int img=0; img<noi; ++img)
        {
            if(finish[img])
                continue;

            // First calculate displaced coordinates
            #pragma omp parallel for
            for(int i=0; i < nos; i++)
            {
                a_coords_displaced[img][i] = a_coords[img][i] + step_size[img] * a_directions[img][i];
            }
            // Get displaced spin directions
            Solver_Kernels::ncg_atlas_to_spins(a_coords_displaced[img], a3_coords[img], *configurations_displaced[img]);
        }
    }

    void ncg_atlas_line_search(  std::vector<std::shared_ptr<vectorfield>> & configurations_displaced, std::vector<vector2field> & a_coords_displaced, std::vector<scalarfield>  & a3_coords, std::vector<vector2field> & a_directions,
                                 std::vector<vectorfield> & forces_displaced, std::vector<vector2field> & a_residuals_displaced, std::vector<std::shared_ptr<Data::Spin_System>> systems,
                                 std::vector<bool> & finish, scalarfield & E0, scalarfield & g0, scalarfield & a_direction_norm, scalarfield & step_size)
    {
        int noi = configurations_displaced.size();
        int nos = configurations_displaced[0]->size();

        for( int img=0; img<noi; ++img )
        {
            fmt::print("line search\n");

            if(finish[img])
                continue;

            // Calculate displaced energy
            scalar Er = systems[img]->hamiltonian->Energy(*configurations_displaced[img]);

            if( std::abs(Er - E0[img]) < 1e-16 )
            {
                fmt::print("Energy too close\n");
                step_size[img] = 0;
                finish[img] = true;
                continue;
            }

            // Calculate displaced residual
            Solver_Kernels::ncg_atlas_residual(a_residuals_displaced[img], a_residuals_displaced[img], *configurations_displaced[img], forces_displaced[img], a3_coords[img] );

            // Calculate displaced directional derivative
            scalar gr = 0;
            #pragma omp parallel for reduction(+:gr)
            for( int i=0; i<nos; ++i )
            {
                gr -= a_residuals_displaced[img][i].dot(a_directions[img][i])/a_direction_norm[img];
            }

            fmt::print("E0   = {}\n", E0[img]);
            fmt::print("Er   = {}\n", Er);
            fmt::print("diff = {}\n", Er-E0[img]);
            fmt::print("g0   = {}\n", g0[img]);
            fmt::print("gr   = {}\n", gr);

            // If wolfe conditions are fulfilled terminate, else suggest new step size
            if( ncg_OSO_wolfe_conditions(E0[img], Er, g0[img], gr, step_size[img]*a_direction_norm[img]) )
            {
                fmt::print("finished\n");
                finish[img] = true;
            } else {
                scalar factor = inexact_line_search(step_size[img]*a_direction_norm[img], E0[img], Er, g0[img], gr);
                fmt::print("continueing\n");
                fmt::print("factor = {}\n", factor);
                if(!std::isnan(factor))
                {
                    step_size[img] *= factor;
                } else {
                    step_size[img] *= 0.5;
                }
            }
        }
    }

    scalar ncg_atlas_norm(const vector2field & a_coords)
    {
        scalar dist = 0;
        #pragma omp parallel for reduction(+:dist)
        for (unsigned int i = 0; i < a_coords.size(); ++i)
            dist += (a_coords[i]).squaredNorm();
        return sqrt(dist);
    }

    scalar ncg_atlas_distance(const vector2field & a_coords1, const vector2field & a_coords2)
    {
        scalar dist = 0;
        #pragma omp parallel for reduction(+:dist)
        for (unsigned int i = 0; i < a_coords2.size(); ++i)
            dist += (a_coords1[i] - a_coords2[i]).squaredNorm() ;
        return sqrt(dist);
    }

    // LBFGS_OSO
    // The "two-loop recursion", see https://en.wikipedia.org/wiki/Limited-memory_BFGS
    // void lbfgs_get_descent_direction(int iteration, int n_updates, vectorfield & a_direction, const vectorfield & residual, const std::vector<vectorfield> & a_updates, const std::vector<vectorfield> & grad_updates, const scalarfield & rho_temp, scalarfield & alpha_temp)
    void lbfgs_get_descent_direction(int iteration, field<int> & n_updates, field<vectorfield> & a_direction, const field<vectorfield> & residual, const field<field<vectorfield>> & a_updates, const field<field<vectorfield>> & grad_updates, const field<scalarfield> & rho_temp, field<scalarfield> & alpha_temp)
    {
        int noi = a_direction.size();
        int nos = a_direction[0].size();

        for(int img=0; img<noi; img++)
        {
            auto & res = residual[img];
            auto & dir = a_direction[img];
            auto & alpha = alpha_temp[img];
            auto & a_up = a_updates[img];
            auto & grad_up = grad_updates[img];
            auto & rho = rho_temp[img];
            auto & n_up = n_updates[img];

            if( n_up == 0 ) // First iteration uses steepest descent
            {
                Vectormath::set_c_a(1, res, dir);
                return;
            }

            Vectormath::set_c_a(1, res, dir); // copy res to dir
            for(int i = iteration; i > iteration - n_up; i--)
            {
                int idx = (i-1) % n_up;
                alpha[idx] = rho[idx] * Vectormath::dot(dir, a_up[idx]);
                Vectormath::add_c_a( -alpha[idx], grad_up[idx], dir );
            }

            int idx_last = (iteration - 1) % n_up;
            scalar top = Vectormath::dot(a_up[idx_last], grad_up[idx_last]);
            scalar bot = Vectormath::dot(grad_up[idx_last], grad_up[idx_last]);
            scalar gamma = top/bot;

            Vectormath::set_c_a(-gamma, dir, dir);
            for(int j = iteration - n_up + 1; j <= iteration; j++)
            {
                int idx = (j-1) % n_up;
                scalar beta = -rho[idx] * Vectormath::dot(grad_up[idx], dir);
                Vectormath::add_c_a( -(alpha[idx] - beta), a_up[idx], dir);
            }
        }
    }

    // LBFGS_OSO
    void lbfgs_get_searchdir(int & local_iter,
           field<scalarfield> & rho,
           field<scalarfield> & alpha,
           field<vectorfield> & q_vec,
           field<vectorfield> & searchdir,
           field<field<vectorfield>> & delta_a,
           field<field<vectorfield>> & delta_grad,
           const field<vectorfield> & grad,
           field<vectorfield> & grad_pr,
           const int num_mem,
           const double maxmove
           )
    {
        int noi = grad.size();
        int nos = grad[0].size();
        int m_index = local_iter % num_mem; // memory index
        int c_ind = 0;
        double scaling = 1.0;

        if (local_iter == 0) {  // gradient descent algorithm
            #pragma omp parallel for
            for(int img=0; img<noi; img++) {
                Vectormath::set_c_a(1.0, grad[img], grad_pr[img]);
                auto & dir = searchdir[img];
                auto & g_cur = grad[img];
                scaling = maximum_rotation(g_cur, maxmove);
                Vectormath::set_c_a(-scaling, g_cur, dir);
                auto & da = delta_a[img];
                auto & dg = delta_grad[img];
                for (int i = 0; i < num_mem; i++){
                    rho[img][i] = 0.0;
                    Vectormath::set_c_a(1.0, {0,0,0}, da[i]);
                    Vectormath::set_c_a(1.0, {0,0,0}, dg[i]);
                }
            }
        } else {
            for (int img=0; img<noi; img++) {
                Vectormath::set_c_a(1, searchdir[img], delta_a[img][m_index]);
                #pragma omp parallel for
                for (int i = 0; i < nos; i++)
                    delta_grad[img][m_index][i] = grad[img][i] - grad_pr[img][i];
            }

            scalar rinv_temp = 0;
            for (int img=0; img<noi; img++) {
                rinv_temp += Vectormath::dot(delta_grad[img][m_index], delta_a[img][m_index]);
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
                Vectormath::set_c_a(1.0, grad[img], q_vec[img]);
            }

            for (int k = num_mem - 1; k > -1; k--)
            {
                c_ind = (k + m_index + 1) % num_mem;
                scalar temp=0;
                for (int img=0; img<noi; img++)
                {
                    temp += Vectormath::dot(delta_a[img][c_ind], q_vec[img]);
                }
                for (int img=0; img<noi; img++)
                {
                    alpha[img][c_ind] = rho[img][c_ind] * temp;
                    Vectormath::add_c_a(-alpha[img][c_ind], delta_grad[img][c_ind], q_vec[img]);
                }
            }

            scalar dy2=0;
            for (int img=0; img<noi; img++)
            {
                dy2 += Vectormath::dot(delta_grad[img][m_index], delta_grad[img][m_index]);
            }
            for (int img=0; img<noi; img++)
            {
                scalar rhody2 = dy2 * rho[img][m_index];
                scalar inv_rhody2 = 0.0;
                if (rhody2 > 1.0e-40)
                    inv_rhody2 = 1.0 / rhody2;
                else
                    inv_rhody2 = 1.0e40;
                Vectormath::set_c_a(inv_rhody2, q_vec[img], searchdir[img]);
            }

            for (int k = 0; k < num_mem; k++)
            {
                if (local_iter < num_mem)
                        c_ind = k;
                    else
                        c_ind = (k + m_index + 1) % num_mem;

                scalar rhopdg = 0;
                for(int img=0; img<noi; img++)
                    rhopdg += Vectormath::dot(delta_grad[img][c_ind], searchdir[img]);

                rhopdg *= rho[0][c_ind];

                for (int img=0; img<noi; img++)
                    Vectormath::add_c_a((alpha[img][c_ind] - rhopdg), delta_a[img][c_ind], searchdir[img]);
            }
            for(int img=0; img<noi; img++)
                scaling = std::min(maximum_rotation(searchdir[img], maxmove), scaling);

            for (int img=0; img<noi; img++)
            {
                Vectormath::set_c_a(1.0, grad[img], grad_pr[img]);
                Vectormath::set_c_a(-scaling, searchdir[img], searchdir[img]);
            }
        }
        local_iter++;
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

    double maximum_rotation(const vectorfield & searchdir, double maxmove){
        int nos = searchdir.size();
        double theta_rms = 0;
        #pragma omp parallel for reduction(+:theta_rms)
        for(int i=0; i<nos; ++i)
            theta_rms += (searchdir[i]).squaredNorm();
        theta_rms = sqrt(theta_rms/nos);
        double scaling = (theta_rms > maxmove) ? maxmove/theta_rms : 1.0;
        return scaling;
    }

    void lbfgs_atlas_get_descent_direction(int iteration, int n_updates, vector2field & a_direction, const vector2field & residual, const std::vector<vector2field> & a_updates, const std::vector<vector2field> & grad_updates, const scalarfield & rho_temp, scalarfield & alpha_temp)
    {
        static auto dot2D = [](const Vector2 & v1, const Vector2 &v2){return v1.dot(v2);};

        if( n_updates <= 3 ) // First iteration uses steepest descent
        {
            Vectormath::set(a_direction, residual, [](Vector2 x){return x;});
            return;
        }

        a_direction = residual; // copy residual to a_direction
        for(int i = iteration; i > iteration - n_updates; i--)
        {
            int idx = (i-1) % n_updates;
            alpha_temp[idx] = rho_temp[idx] * Vectormath::reduce(a_direction, a_updates[idx], dot2D);
            // Vectormath::add_c_a( -alpha_temp[idx], grad_updates[idx], a_direction );
            Vectormath::set( a_direction, grad_updates[idx], [&alpha_temp, idx](const Vector2 & v){return alpha_temp[idx] * v;} );
        }

        int idx_last = (iteration - 1) % n_updates;
        scalar top = Vectormath::reduce(a_updates[idx_last], grad_updates[idx_last], dot2D);
        scalar bot = Vectormath::reduce(grad_updates[idx_last], grad_updates[idx_last], dot2D);
        scalar gamma = top/bot;

        // Vectormath::set_c_a(-gamma, a_direction, a_direction);
        Vectormath::set(a_direction, a_direction, [gamma](const Vector2 & v){return -gamma*v;});

        for(int j = iteration - n_updates + 1; j <= iteration; j++)
        {
            int idx = (j-1) % n_updates;
            scalar beta = -rho_temp[idx] * Vectormath::reduce(grad_updates[idx], a_direction, dot2D);
            Vectormath::set( a_direction, a_updates[idx], [idx, &alpha_temp, beta](const Vector2 & v){return -(alpha_temp[idx] - beta) * v;});
        }
    }

    void atlas_rotate(std::vector<std::shared_ptr<vectorfield>> & configurations, field <vector2field> & a_coords, field<scalarfield> & a3_coords, std::vector<vector2field> & searchdir)
    {
        int noi = configurations.size();
        int nos = configurations[0]->size();
        for(int img=0; img<noi; img++ )
        {
            #pragma omp parallel for
            for(int i=0; i < nos; i++)
            {
                a_coords[img][i] += searchdir[img][i];
            }
            // Get displaced spin directions
            Solver_Kernels::ncg_atlas_to_spins(a_coords[img], a3_coords[img], *configurations[img]);
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

    void lbfgs_atlas_transform_direction(field<std::shared_ptr<vectorfield>> & configurations, field<vector2field> & a_coords, field<scalarfield> & a3_coords, field<field<vector2field>> & atlas_updates, field<field<vector2field>> & grad_updates, field<vector2field> & searchdir, field<vector2field> & grad_pr, field<scalarfield> & rho)
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
                auto &       a = a_coords[img][i];
                auto &      a3 = a3_coords[img][i];

                if( s[2]*a3 < 0 )
                {
                    // Transform coordinates to optimal map
                    a3 = (s[2] > 0) ? 1 : -1;
                    a[0] = s[0] / (1 + s[2]*a3);
                    a[1] = s[1] / (1 + s[2]*a3);

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