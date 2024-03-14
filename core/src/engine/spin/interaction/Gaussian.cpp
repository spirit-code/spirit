#include <engine/spin/interaction/Gaussian.hpp>

namespace Engine
{

namespace Spin
{

namespace Interaction
{

template<>
scalar Gaussian::Energy::operator()( const Index & index, const vectorfield & spins ) const
{
    scalar result = 0;

    if( !index.has_value() )
        return result;

    const int ispin = *index;

    for( unsigned int igauss = 0; igauss < data.amplitude.size(); ++igauss )
    {
        // Distance between spin and gaussian center
        scalar l = 1 - data.center[igauss].dot( spins[ispin] );
        result += data.amplitude[igauss] * std::exp( -l * l / ( 2.0 * data.width[igauss] * data.width[igauss] ) );
    };
    return result;
};

template<>
Vector3 Gaussian::Gradient::operator()( const Index & index, const vectorfield & spins ) const
{
    Vector3 result = Vector3::Zero();

    if( !index.has_value() )
        return result;

    const int ispin = *index;
    // Calculate gradient
    for( unsigned int i = 0; i < data.amplitude.size(); ++i )
    {
        // Scalar product of spin and gaussian center
        scalar l = 1 - data.center[i].dot( spins[ispin] );
        // Prefactor
        scalar prefactor = data.amplitude[i] * std::exp( -l * l / ( 2.0 * data.width[i] * data.width[i] ) ) * l
                           / ( data.width[i] * data.width[i] );
        // Gradient contribution
        result += prefactor * data.center[i];
    }
    return result;
}

} // namespace Interaction

} // namespace Spin

} // namespace Engine
