#include <engine/Vectormath.hpp>
#include <engine/interaction/ABC.hpp>
#include <utility/Exception.hpp>
#include <utility/Logging.hpp>

using namespace Utility;
using namespace Data;

namespace Engine
{

namespace Interaction
{

void ABC::updateFromGeometry( const Geometry * geometry ){};

scalar ABC::Energy( const vectorfield & spins )
{
    auto nos = spins.size();
    if( energy_per_spin.size() != nos )
        energy_per_spin = scalarfield( nos, 0 );
    else
        Vectormath::fill( energy_per_spin, 0 );

    Energy_per_Spin( spins, energy_per_spin );

    return Vectormath::sum( energy_per_spin );
}

void ABC::Hessian( const vectorfield & spins, MatrixX & hessian )
{
    this->Hessian_FD( spins, hessian );
}

std::size_t ABC::Sparse_Hessian_Size_per_Cell() const
{
    return 0;
};

void ABC::Hessian_FD( const vectorfield & spins, MatrixX & hessian )
{
    // This is a regular finite difference implementation (probably not very efficient)
    // using the differences between gradient values (not function)
    // see https://v8doc.sas.com/sashtml/ormp/chap5/sect28.htm

    std::size_t nos = spins.size();

    vectorfield spins_pi( nos );
    vectorfield spins_mi( nos );
    vectorfield spins_pj( nos );
    vectorfield spins_mj( nos );

    spins_pi = spins;
    spins_mi = spins;
    spins_pj = spins;
    spins_mj = spins;

    vectorfield grad_pi( nos, Vector3::Zero() );
    vectorfield grad_mi( nos, Vector3::Zero() );
    vectorfield grad_pj( nos, Vector3::Zero() );
    vectorfield grad_mj( nos, Vector3::Zero() );

    for( std::size_t i = 0; i < nos; ++i )
    {
        for( std::size_t j = 0; j < nos; ++j )
        {
#pragma unroll
            for( std::uint8_t alpha = 0; alpha < 3; ++alpha )
            {
#pragma unroll
                for( std::uint8_t beta = 0; beta < 3; ++beta )
                {
                    // Displace
                    spins_pi[i][alpha] += delta;
                    spins_mi[i][alpha] -= delta;
                    spins_pj[j][beta] += delta;
                    spins_mj[j][beta] -= delta;

                    Vectormath::fill( grad_pi, Vector3::Zero() );
                    Vectormath::fill( grad_mi, Vector3::Zero() );
                    Vectormath::fill( grad_pj, Vector3::Zero() );
                    Vectormath::fill( grad_mj, Vector3::Zero() );

                    // Calculate Hessian component
                    this->Gradient( spins_pi, grad_pi );
                    this->Gradient( spins_mi, grad_mi );
                    this->Gradient( spins_pj, grad_pj );
                    this->Gradient( spins_mj, grad_mj );

                    hessian( 3 * i + alpha, 3 * j + beta )
                        = 0.25 / delta
                          * ( grad_pj[i][alpha] - grad_mj[i][alpha] + grad_pi[j][beta] - grad_mi[j][beta] );

                    // Un-Displace
                    spins_pi[i][alpha] -= delta;
                    spins_mi[i][alpha] += delta;
                    spins_pj[j][beta] -= delta;
                    spins_mj[j][beta] += delta;
                }
            }
        }
    }
}

void ABC::Gradient( const vectorfield & spins, vectorfield & gradient )
{
    this->Gradient_FD( spins, gradient );
}

void ABC::Gradient_FD( const vectorfield & spins, vectorfield & gradient )
{
    std::size_t nos = spins.size();

    // Calculate finite difference
    vectorfield spins_plus( nos );
    vectorfield spins_minus( nos );

    spins_plus  = spins;
    spins_minus = spins;

    for( std::size_t i = 0; i < nos; ++i )
    {
        for( std::uint8_t dim = 0; dim < 3; ++dim )
        {
            // Displace
            spins_plus[i][dim] += delta;
            spins_minus[i][dim] -= delta;

            // Calculate gradient component
            scalar E_plus    = this->Energy( spins_plus );
            scalar E_minus   = this->Energy( spins_minus );
            gradient[i][dim] = 0.5 * ( E_plus - E_minus ) / delta;

            // Un-Displace
            spins_plus[i][dim] -= delta;
            spins_minus[i][dim] += delta;
        }
    }
}

scalar ABC::Energy_Single_Spin( const int ispin, const vectorfield & spins )
{
    Log( Log_Level::Error, Log_Sender::All,
         fmt::format(
             "Interaction: {} uses the highly inefficient fallback implementation for Energy_Single_Spin()!",
             this->Name() ) );
    if( this->spin_order().has_value() )
    {
        energy_per_spin[ispin] = 0;
        Energy_per_Spin( spins, this->energy_per_spin );
        return this->spin_order().value() * energy_per_spin[ispin];
    }
    else
    {
        spirit_throw(
            Exception_Classifier::Not_Implemented, Log_Level::Error,
            fmt::format(
                "Cannot use fallback Interaction::ABC::Energy_Single_Spin() without a valid spin_order(), "
                "please implement a proper Energy_Single_Spin() method for your subclass: \"{}\"",
                this->Name() ) );
    }
}

} // namespace Interaction

} // namespace Engine
