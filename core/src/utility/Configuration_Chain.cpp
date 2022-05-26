#include <data/Spin_System.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Backend_par.hpp>
#include <Eigen/Geometry>

#include <utility/Configuration_Chain.hpp>
#include <utility/Configurations.hpp>
#include <utility/Constants.hpp>
#include <utility/Logging.hpp>

#include <Eigen/Dense>

#include <iostream>
#include <random>
#include <string>
#include <vector>


namespace Utility
{
namespace Configuration_Chain
{

void Add_Noise_Temperature( std::shared_ptr<Data::Spin_System_Chain> c, int idx_1, int idx_2, scalar temperature )
{
    for( int img = idx_1 + 1; img <= idx_2 - 1; ++img )
    {
        Configurations::Add_Noise_Temperature( *c->images[img], temperature, img );
    }
}

void Homogeneous_Rotation( std::shared_ptr<Data::Spin_System_Chain> c, int idx_1, int idx_2 )
{
    auto & spins_1 = *c->images[idx_1]->spins;
    auto & spins_2 = *c->images[idx_2]->spins;

    scalar angle, rot_angle;
    Vector3 rot_axis;
    Vector3 ex = { 1, 0, 0 };
    Vector3 ey = { 0, 1, 0 };

    bool antiparallel = false;
    for( int i = 0; i < c->images[0]->nos; ++i )
    {
        rot_angle = Engine::Vectormath::angle( spins_1[i], spins_2[i] );
        rot_axis  = spins_1[i].cross( spins_2[i] ).normalized();

        // If spins are antiparallel, we choose an arbitrary rotation axis
        if( std::abs( rot_angle - Constants::Pi ) < 1e-4 )
        {
            antiparallel = true;
            if( std::abs( spins_1[i].dot( ex ) ) - 1 > 1e-4 )
                rot_axis = ex;
            else
                rot_axis = ey;
        }

        // If they are not strictly parallel we can rotate
        if( rot_angle > 1e-8 )
        {
            for( int img = idx_1 + 1; img < idx_2; ++img )
            {
                angle = rot_angle * scalar( img - idx_1 ) / scalar( idx_2 - idx_1 );
                Engine::Vectormath::rotate( spins_1[i], rot_axis, angle, ( *c->images[img]->spins )[i] );
            }
        }
        // Otherwise we simply leave the spin untouched
        else
        {
            for( int img = idx_1 + 1; img < idx_2; ++img )
            {
                ( *c->images[img]->spins )[i] = spins_1[i];
            }
        }
    }
    if( antiparallel )
        Log( Log_Level::Warning, Log_Sender::All,
             "For the interpolation of antiparallel spins an arbitrary rotation axis has been chosen." );
}

// Shift dimer
void Dimer_Shift( std::shared_ptr<Data::Spin_System_Chain> c, bool invert )
{
    if(c->noi != 2)
    {
        Log( Log_Level::Error, Log_Sender::All, "Not a Dimer." );
    }

    // auto & spins_left  = *c->images[0]->spins;
    // auto & spins_right = *c->images[1]->spins;

    int nos = c->images[0]->nos;

    // clang-format off
    Engine::Backend::par::apply(nos,
        [
            spins_left  = c->images[0]->spins->data(),
            spins_right = c->images[c->noi-1]->spins->data(),
            invert
        ] SPIRIT_LAMBDA ( int idx)
        {
            Vector3 axis = spins_left[idx].cross(spins_right[idx]);
            scalar angle = acos(spins_left[idx].dot(spins_right[idx]));

            // Rotation matrix that rotates spin_left to spin_right
            Matrix3 rotation_matrix = Eigen::AngleAxis<scalar>(angle, axis.normalized()).toRotationMatrix();

            if (std::abs(angle) < 1e-6 || std::isnan(angle)) // Angle can become nan for collinear spins
                rotation_matrix = Matrix3::Identity();

            if(!invert)
            {
                spins_left[idx] = spins_right[idx];
                spins_right[idx] = rotation_matrix * spins_right[idx];
            } else {
                spins_right[idx] = spins_left[idx];
                spins_left[idx] = rotation_matrix.transpose() * spins_left[idx];
            }
        }
    );
}

} // namespace Configuration_Chain
} // namespace Utility
