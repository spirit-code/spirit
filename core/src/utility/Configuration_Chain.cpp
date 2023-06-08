#include <data/Spin_System.hpp>
#include <engine/Vectormath.hpp>
#include <utility/Configuration_Chain.hpp>
#include <utility/Configurations.hpp>
#include <utility/Constants.hpp>
#include <utility/Logging.hpp>

#include <Eigen/Dense>

#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace Spirit::Utility::Configuration_Chain
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

} // namespace Spirit::Utility::Configuration_Chain
