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
    void ncg_OSO_residual( vectorfield & a_residuals, vectorfield & a_residuals_last, const vectorfield & spins, const vectorfield & a_coords,
                       const vectorfield & forces, bool approx)
    {
        using cpx = std::complex<scalar>;

        if(approx) // Use approximate gradient
        {
            #pragma omp parallel for
            for( int i =0; i<spins.size(); i++)
            {
                a_residuals_last[i] = a_residuals[i];
                Vector3 temp = -spins[i].cross(forces[i]);
                a_residuals[i][0] = temp[2];
                a_residuals[i][1] = -temp[1];
                a_residuals[i][2] = temp[0];
            }
        } else { // Use exact gradient
            //todo
            // const scalar & a = a_coords[i][0], b = a_coords[i][1], c = a_coords[i][2];
            // scalar x = sqrt(a*a + b*b + c*c);
            // cpx lambda2 = cpx(0,-x);
            // cpx lambda3 = cpx(0,x);
            // Eigen::Matrix<3,3> V;
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

    void ncg_OSO_displace( std::vector<std::shared_ptr<vectorfield>> & configurations_displaced, std::vector<std::shared_ptr<vectorfield>> & reference_configurations, std::vector<vectorfield> & a_coords,
                           std::vector<vectorfield> & a_coords_displaced, std::vector<vectorfield> & a_directions, std::vector<bool> finish, scalarfield step_size )
    {
        int noi = configurations_displaced.size();
        int nos = configurations_displaced[0]->size();

        for(int img=0; img<noi; ++img)
        {
            if(finish[img])
                continue;

            // First calculate displaced coordinates
            for(int i=0; i < nos; i++)
            {
                a_coords_displaced[img][i] = a_coords[img][i] + step_size[img] * a_directions[img][i];
            }
            // Get displaced spin directions
            Solver_Kernels::ncg_OSO_a_to_spins(*configurations_displaced[img], a_coords_displaced[img], *reference_configurations[img]);
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
            Solver_Kernels::ncg_OSO_residual(a_residuals_displaced[img], a_residuals_displaced[img], *configurations_displaced[img], a_coords_displaced[img], forces_displaced[img]);

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
                if(!isnan(factor))
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

            J(0,0) =  s[1]*s[1]  + s[2]*(s[2] + a3);
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

    void ncg_atlas_check_coordinates(const vectorfield & spins, vector2field & a_coords, scalarfield & a3_coords, vector2field & a_directions)
    {
         // Check if we need to reset the maps
        bool reset = false;

        #pragma omp parallel for
        for( int i=0; i<spins.size(); i++ )
        {
            // If for one spin the z component deviates too much from the pole we perform a reset for *all* spins
            // Note: I am not sure why we reset for all spins ... but this is in agreement with the method of F. Rybakov
            if(spins[i][2]*a3_coords[i] < -0.5)
            {
                reset = true;
                break;
            }
        }

        if(reset)
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
    }

    void ncg_atlas_displace( std::vector<std::shared_ptr<vectorfield>> & configurations_displaced, std::vector<vector2field> & a_coords, std::vector<scalarfield> & a3_coords,
                             std::vector<vector2field> & a_coords_displaced, std::vector<vector2field> & a_directions, std::vector<bool> finish, scalarfield step_size )
    {
        int noi = configurations_displaced.size();
        int nos = configurations_displaced[0]->size();

        #pragma omp parallel for
        for(int img=0; img<noi; ++img)
        {
            if(finish[img])
                continue;

            // First calculate displaced coordinates
            for(int i=0; i < nos; i++)
            {
                a_coords_displaced[img][i] = a_coords[img][i] + step_size[img] * a_directions[img][i];
            }
            // Get displaced spin directions
            Solver_Kernels::ncg_atlas_to_spins(a_coords_displaced[img], a3_coords[img], *configurations_displaced[img]);
        }
    }

    bool ncg_atlas_line_search(  std::vector<std::shared_ptr<vectorfield>> & configurations_displaced, std::vector<vector2field> & a_coords_displaced, std::vector<scalarfield>  & a3_coords, std::vector<vector2field> & a_directions,
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
                if(!isnan(factor))
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

    // LBFGS
    // The "two-loop recursion", see https://en.wikipedia.org/wiki/Limited-memory_BFGS
    void lbfgs_get_descent_direction(int iteration, int n_lbfgs_memory, vectorfield & a_direction, const vectorfield & residual, const std::vector<vectorfield> & spin_updates, const std::vector<vectorfield> & grad_updates, const scalarfield & rho_temp, scalarfield & alpha_temp)
    {
        if( iteration <= 1 ) // First iteration uses steepest descent
        {
            Vectormath::set_c_a(1, residual, a_direction);
            return;
        }

        int n_updates = std::min(n_lbfgs_memory, iteration);

        Vectormath::set_c_a(1, residual, a_direction); // copy residual to a_direction
        for(int i = iteration; i > iteration - n_updates; i--)
        {
            int idx = (i-1) % n_lbfgs_memory;
            alpha_temp[idx] = rho_temp[idx] * Vectormath::dot(a_direction, spin_updates[idx]);
            Vectormath::add_c_a( -alpha_temp[idx], grad_updates[idx], a_direction );
        }

        int idx_last = (iteration - 1) % n_lbfgs_memory;
        scalar top = Vectormath::dot(spin_updates[idx_last], grad_updates[idx_last]);
        scalar bot = Vectormath::dot(grad_updates[idx_last], grad_updates[idx_last]);
        scalar gamma = top/bot;

        Vectormath::set_c_a(-gamma, a_direction, a_direction);
        for(int j = iteration - n_updates + 1; j <= iteration; j++)
        {
            int idx = (j-1) % n_updates;
            scalar beta = -rho_temp[idx] * Vectormath::dot(grad_updates[idx], a_direction);
            Vectormath::add_c_a( -(alpha_temp[idx] - beta), spin_updates[idx], a_direction);
        }
    }

    #endif
}
}