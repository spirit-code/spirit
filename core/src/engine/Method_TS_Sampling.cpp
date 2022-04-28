#include <engine/Method_TS_Sampling.hpp>
#include <engine/Backend_par.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Eigenmodes.hpp>
#include <engine/Manifoldmath.hpp>
#include <utility/Constants.hpp>
#include <data/Parameters_Method.hpp>
#include <Eigen/Geometry>
#include <fmt/format.h>
#include <fmt/ostream.h>

using namespace Utility;

namespace Engine
{

Method_TS_Sampling::Method_TS_Sampling( std::shared_ptr<Data::Spin_System> system, int idx_img, int idx_chain ) : Method_MC(system, idx_img, idx_chain)
{
    transition_plane_normal = vectorfield(this->nos, {1,0,0});
    rejected                = std::deque<int>();
    for(int i=0; i<50; i++)
    {
        rejected.push_back(0);
        rejected.push_back(1);
    }
    distribution_idx = std::uniform_int_distribution<>( 0, this->nos - 1 );
    Method_TS_Sampling::Compute_Transition_Plane_Normal( );
}

void Method_TS_Sampling::Set_Transition_Plane_Normal( vectorfield & spins_minimum, vectorfield & unstable_mode )
{
    for( int idx = 0; idx < this->nos; ++idx )
    {
        this->transition_plane_normal[idx] = unstable_mode[idx] - spins_minimum[idx].dot(unstable_mode[idx]) * spins_minimum[idx];
    }
    Manifoldmath::normalize(this->transition_plane_normal);
}

void Method_TS_Sampling::Compute_Transition_Plane_Normal( )
{
    auto & spins_initial = *this->systems[0]->spins;
    Eigenmodes::Calculate_Eigenmodes( this->systems[0], 0, 0 );

    Log(Utility::Log_Level::Info, Utility::Log_Sender::MC, "Trying to compute transition plane normal");
    Set_Transition_Plane_Normal( spins_initial, *this->systems[0]->modes[0] );
}

void Method_TS_Sampling::Iteration()
{
    // Temporaries
    auto & spins_old = *this->systems[0]->spins;
    scalar Eold = this->systems[0]->hamiltonian->Energy( spins_old );
    Vectormath::set_c_a(1, spins_old, this->spins_new);

    auto indices   = std::vector<int>();
    auto old_spins = std::vector<Vector3>();

    auto distribution = std::uniform_real_distribution<scalar>( 0, 1 );
    for( int idx = 0; idx < this->nos; ++idx )
    {
        Method_MC::Displace_Spin(idx, spins_new, distribution, indices, old_spins);
    }

    // Compute correction
    for(int n_iter=0; n_iter<10; n_iter++)
    {
        scalar dot_product = Vectormath::dot( spins_new, transition_plane_normal ) / nos;
        Manifoldmath::project_orthogonal( spins_new, transition_plane_normal, dot_product );
        Vectormath::normalize_vectors( spins_new );
        if(dot_product < 1e-15)
            break;
    }

    scalar Enew = this->systems[0]->hamiltonian->Energy( spins_new );
    scalar Ediff = Enew - Eold;

    // Metropolis criterion: reject the step if energy rose
    bool reject = false;
    const scalar kB_T = Utility::Constants::k_B * this->parameters_mc->temperature;
    if( Ediff > 1e-14 )
    {
        if( this->parameters_mc->temperature < 1e-12 )
        {
            reject = true;
        }
        else
        {
            // Exponential factor
            scalar exp_ediff = std::exp( -Ediff / kB_T );
            // Metropolis random number
            scalar x_metropolis = distribution( this->parameters_mc->prng );
            // Only reject if random number is larger than exponential
            if( exp_ediff < x_metropolis )
            {
                reject = true;
            }
        }
    }

    // Record rejections
    rejected.pop_front();
    rejected.push_back(int(reject));

    n_rejected = 0;
    for(auto t : rejected)
    {
        n_rejected += t;
    }

    scalar diff = 0.01;
    // Cone angle feedback algorithm
    if( this->parameters_mc->metropolis_step_cone && this->parameters_mc->metropolis_cone_adaptive )
    {
        this->acceptance_ratio_current = 1 - (scalar)this->n_rejected / (scalar)rejected.size();

        if( ( this->acceptance_ratio_current < this->parameters_mc->acceptance_ratio_target )
            && ( this->cone_angle > diff ) )
            this->cone_angle -= diff;

        if( ( this->acceptance_ratio_current > this->parameters_mc->acceptance_ratio_target )
            && ( this->cone_angle < Constants::Pi - diff ) )
            this->cone_angle += diff;

        this->parameters_mc->metropolis_cone_angle = this->cone_angle * 180.0 / Constants::Pi;
    }

    if(!reject)
    {
        Vectormath::set_c_a( 1, this->spins_new, spins_old ); // Only copy out, if not rejected
    }
}

// void Method_TS_Sampling::Displace_Spin(int ispin, vectorfield & spins_new, std::uniform_real_distribution<scalar> & distribution, std::vector<int> & changed_indices, vectorfield & old_spins)
// {
//     Method_MC::Displace_Spin( ispin, spins_new, distribution, changed_indices, old_spins);

//     for(int n_iter=0; n_iter<10; n_iter++)
//     {
//         scalar dot_product = Vectormath::dot( spins_new, transition_plane_normal ) / nos;
//         Manifoldmath::project_orthogonal( spins_new, transition_plane_normal, dot_product );
//         Vectormath::normalize_vectors( spins_new );

//         fmt::print( "dot_product = {}\n", dot_product );
//         // debug_prints(spins_new, transition_plane_normal);
//     }

    // fmt::print(" ======\n");


    // // Debug
    // debug_prints(spins_new, transition_plane_normal);
    // // Perform the normal displacement
    // scalar delta_scalar_product = (spins_new[ispin] - old_spins[0]).dot(transition_plane_normal[ispin]);

    // // fmt::print("delta_scalar_product {}\n", delta_scalar_product);

    // // Correct displacement
    // // spin new[ispin2] needs to compensate for change in total scalar product
    // int ispin2 = distribution_idx( this->parameters_mc->prng );
    // // ispin2 = (ispin + 1) % 2;
    // // fmt::print( "ispin = {}, ispin2 = {}\n", ispin, ispin2);

    // scalar current_angle = std::acos( spins_new[ispin2].dot( transition_plane_normal[ispin2].normalized() ) );

    // scalar delta_angle = 0;
    // if(std::abs(delta_scalar_product) > 1e-8) // Only compensate if there is a change to the scalarproduct
    // {
    //     delta_angle = std::acos( std::cos(current_angle) - delta_scalar_product / transition_plane_normal[ispin2].norm() ) - current_angle;
    //     if(std::isnan(delta_angle))
    //     {
    //         // Compensation not possible

    //     }
    // }
    // // fmt::print( "   delta_angle                      = {}\n",      delta_angle);
    // // fmt::print( "   cos(current_angle)               = {}\n",      std::cos(current_angle) );
    // // fmt::print( "   cos(current_angle + delta_angle) = {} = {}\n", std::cos(current_angle + delta_angle), delta_scalar_product );
    // // fmt::print( "prod before = {}\n", spins_new[ispin2].dot(transition_plane_normal[ispin2]));

    // const Vector3 axis      = spins_new[ispin2].cross(transition_plane_normal[ispin2].normalized());
    // Matrix3 rotation_matrix = Eigen::AngleAxis<scalar>(-delta_angle, axis.normalized()).toRotationMatrix();
    // old_spins.push_back(spins_new[ispin2]);

    // spins_new[ispin2]       = rotation_matrix * spins_new[ispin2];

    // // fmt::print("prod after = {}\n", spins_new[ispin2].dot(transition_plane_normal[ispin2]));
    // // fmt::print("spins[ispin2] after {}\n", spins_new[ispin2].transpose());
    // // fmt::print("current_angle {}\ndelta_angle {}\naxis {}\n", current_angle, delta_angle, axis.transpose());

    // changed_indices = {ispin, ispin2};
    // fmt::print(" -------------------------------------- \n");
    // return 100;
    // return Ediff;
// }

// Method name as string
std::string Method_TS_Sampling::Name()
{
    return "TS_Sampling";
}

}