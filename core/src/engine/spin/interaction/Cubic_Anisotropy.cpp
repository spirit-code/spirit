#include <engine/spin/interaction/Cubic_Anisotropy.hpp>
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
scalar Cubic_Anisotropy::Energy::operator()( const Index & index, const vectorfield & spins ) const
{
    using std::pow;
    scalar result = 0;
    if( !index.has_value() )
        return result;

    const auto & [ispin, iani] = *index;
    return -0.5 * data.magnitudes[iani]
           * ( pow( spins[ispin][0], 4.0 ) + pow( spins[ispin][1], 4.0 ) + pow( spins[ispin][2], 4.0 ) );
};

template<>
Vector3 Cubic_Anisotropy::Gradient::operator()( const Index & index, const vectorfield & spins ) const
{
    using std::pow;
    Vector3 result = Vector3::Zero();
    if( !index.has_value() )
        return result;

    const auto & [ispin, iani] = *index;

    for( int icomp = 0; icomp < 3; ++icomp )
    {
        result[icomp] = -2.0 * data.magnitudes[iani] * pow( spins[ispin][icomp], 3.0 );
    }
    return result;
}

} // namespace Interaction

} // namespace Spin

} // namespace Engine
