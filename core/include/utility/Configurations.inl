#pragma once

#include <engine/Backend.hpp>
#include <engine/Vectormath.hpp>
#include <utility/Configurations.hpp>
#include <utility/Constants.hpp>

namespace Utility
{

namespace Configurations
{

template<typename RandomFunc>
void Random_Sphere( vectorfield & spins, const Data::Geometry & geometry, RandomFunc & prng, filterfunction filter )
{
    auto & positions = geometry.positions;

    auto distribution = std::uniform_real_distribution<scalar>( -1, 1 );
    for( unsigned int iatom = 0; iatom < spins.size(); ++iatom )
    {
        if( filter( positions[iatom] ) )
        {
            Engine::Vectormath::get_random_vector_unitsphere( distribution, prng, spins[iatom] );
        }
    }
}

template<typename RandomFunc>
void Random_Cube( vectorfield & field, const Data::Geometry & geometry, RandomFunc & prng, filterfunction filter )
{
    auto & positions = geometry.positions;

    auto distribution = std::uniform_real_distribution<scalar>( -1, 1 );
    for( unsigned int iatom = 0; iatom < field.size(); ++iatom )
    {
        if( filter( positions[iatom] ) )
        {
            Engine::Vectormath::get_random_vector( distribution, prng, field[iatom] );
        }
    }
}

template<typename RandomFunc>
void Add_Noise_Temperature_Sphere(
    vectorfield & spins, const Data::Geometry & geometry, scalar temperature, RandomFunc & prng, filterfunction filter )
{
    if( temperature == 0.0 )
        return;

    auto & positions = geometry.positions;
    vectorfield xi( spins.size() );
    intfield mask;

    filter_to_mask( positions, filter, mask );

    scalar epsilon = std::sqrt( temperature * Constants::k_B );

    Engine::Vectormath::get_random_vectorfield_unitsphere( prng, xi );
    Engine::Backend::for_each_n(
        SPIRIT_PAR Engine::Backend::make_zip_iterator( spins.begin(), xi.begin(), mask.begin() ), spins.size(),
        Engine::Backend::make_zip_function(
            [epsilon] SPIRIT_LAMBDA( Vector3 & n, const Vector3 & xi, const bool pred )
            {
                if( pred )
                    n = ( n + epsilon * xi ).normalized();
            } ) );
}

template<typename RandomFunc>
void Add_Noise_Temperature_Cube(
    vectorfield & field, const Data::Geometry & geometry, scalar temperature, RandomFunc & prng, filterfunction filter )
{
    // TODO(spin-lattice): decide how to implement independent noise for Spin, Displacment & Momentum
    if( temperature == 0.0 )
        return;

    auto & positions = geometry.positions;
    vectorfield xi( field.size() );
    intfield mask;

    filter_to_mask( positions, filter, mask );

    scalar epsilon = std::sqrt( temperature * Constants::k_B );

    Engine::Vectormath::get_random_vectorfield( prng, xi );
    Engine::Backend::for_each_n(
        SPIRIT_PAR Engine::Backend::make_zip_iterator( field.begin(), xi.begin(), mask.begin() ), field.size(),
        Engine::Backend::make_zip_function(
            [epsilon] SPIRIT_LAMBDA( Vector3 & n, const Vector3 & xi, const bool pred )
            {
                if( pred )
                    n += epsilon * xi;
            } ) );
}

} // namespace Configurations

} // namespace Utility
