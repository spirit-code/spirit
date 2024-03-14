#include <engine/spin/interaction/Anisotropy.hpp>
#include <utility/Constants.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

namespace Engine
{

namespace Spin
{

namespace Interaction
{

template<>
scalar Anisotropy::Energy::operator()( const Index & index, const vectorfield & spins ) const
{
    if( index.has_value() )
    {
        const auto & [ispin, iani] = *index;
        const auto d               = data.normals[iani].dot( spins[ispin] );
        return -data.magnitudes[iani] * d * d;
    }
    else
    {
        return 0;
    }
}

template<>
Vector3 Anisotropy::Gradient::operator()( const Index & index, const vectorfield & spins ) const
{
    if( index.has_value() )
    {
        const auto & [ispin, iani] = *index;
        return -2.0 * data.magnitudes[iani] * data.normals[iani]
               * data.normals[iani].dot( spins[ispin] );
    }
    else
    {
        return Vector3::Zero();
    }
};

} // namespace Interaction

} // namespace Spin

} // namespace Engine
