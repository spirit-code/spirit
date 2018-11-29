#include <utility/Configuration_Chain.hpp>
#include <utility/Configurations.hpp>
#include <utility/Constants.hpp>
#include <data/Spin_System.hpp>
#include <engine/Vectormath.hpp>

#include <Eigen/Dense>

#include <random>
#include <iostream>
#include <string>
#include <vector>


namespace Utility
{
    namespace Configuration_Chain
    {
        void Add_Noise_Temperature(std::shared_ptr<Data::Spin_System_Chain> c, int idx_1, int idx_2, scalar temperature)
        {
            for (int img = idx_1 + 1; img <= idx_2 - 1; ++img)
            {
                Configurations::Add_Noise_Temperature(*c->images[img], temperature, img);
            }
        }

        void Homogeneous_Rotation(std::shared_ptr<Data::Spin_System_Chain> c, int idx_1, int idx_2)
        {
            auto& spins_1 = *c->images[idx_1]->spins;
            auto& spins_2 = *c->images[idx_2]->spins;

            scalar angle, rot_angle;
            Vector3 rot_axis;

            for( int i = 0; i < c->images[0]->nos; ++i )
            {
                rot_angle = Engine::Vectormath::angle(spins_1[i], spins_2[i]);
                rot_axis = spins_1[i].cross(spins_2[i]).normalized();

                // If they are not strictly parallel we can rotate
                if( rot_angle > 1e-8 )
                {
                    for( int img = idx_1+1; img < idx_2; ++img )
                    {
                        angle = rot_angle*scalar(img-idx_1)/scalar(idx_2 - idx_1);
                        Engine::Vectormath::rotate(spins_1[i], rot_axis, angle, (*c->images[img]->spins)[i]);
                    }
                }
                // Otherwise we simply leave the spin untouched
                else
                {
                    for( int img = idx_1+1; img < idx_2; ++img )
                    {
                        (*c->images[img]->spins)[i] = spins_1[i];
                    }
                }
            }
        }
    }//end namespace Configuration_Chain
}//end namespace Utility