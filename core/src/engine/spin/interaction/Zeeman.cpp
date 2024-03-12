#include <engine/spin/interaction/Zeeman.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

namespace Engine
{

namespace Spin
{

namespace Interaction
{

template<>
scalar Zeeman::Energy::operator()( const Index & index, const vectorfield & spins ) const
{
    if( cache.geometry == nullptr )
        return 0;

    if( index.has_value() && *index >= 0 )
    {
        const auto & mu_s  = cache.geometry->mu_s;
        const auto & ispin = *index;
        return -mu_s[ispin] * data.external_field_magnitude * data.external_field_normal.dot( spins[ispin] );
    }
    else
        return 0;
};

template<>
Vector3 Zeeman::Gradient::operator()( const Index & index, const vectorfield & ) const
{
    if( cache.geometry == nullptr )
        return Vector3::Zero();

    if( index.has_value() && *index >= 0 )
    {
        const auto & mu_s  = cache.geometry->mu_s;
        const auto & ispin = *index;
        return -mu_s[ispin] * data.external_field_magnitude * data.external_field_normal;
    }
    else
        return Vector3::Zero();
}

} // namespace Interaction

} // namespace Spin

} // namespace Engine
