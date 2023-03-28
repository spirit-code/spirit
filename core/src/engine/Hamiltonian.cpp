#include <engine/Hamiltonian.hpp>
#include <engine/Vectormath.hpp>
#include <utility/Exception.hpp>
#include <utility/Logging.hpp>

using namespace Utility;

namespace Engine
{

Hamiltonian::Hamiltonian( intfield boundary_conditions )
        : boundary_conditions{ std::move( boundary_conditions ) }, energy_contributions_per_spin_{}, delta_{ 1e-3 }
{
}

void Hamiltonian::Hessian_FD( const vectorfield & spins, MatrixX & hessian )
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

    vectorfield grad_pi( nos );
    vectorfield grad_mi( nos );
    vectorfield grad_pj( nos );
    vectorfield grad_mj( nos );

    for( std::size_t i = 0; i < nos; ++i )
    {
        for( std::size_t j = 0; j < nos; ++j )
        {
            for( std::uint8_t alpha = 0; alpha < 3; ++alpha )
            {
                for( std::uint8_t beta = 0; beta < 3; ++beta )
                {
                    // Displace
                    spins_pi[i][alpha] += delta_;
                    spins_mi[i][alpha] -= delta_;
                    spins_pj[j][beta] += delta_;
                    spins_mj[j][beta] -= delta_;

                    // Calculate Hessian component
                    Gradient( spins_pi, grad_pi );
                    Gradient( spins_mi, grad_mi );
                    Gradient( spins_pj, grad_pj );
                    Gradient( spins_mj, grad_mj );

                    hessian( 3 * i + alpha, 3 * j + beta )
                        = 0.25 / delta_
                          * ( grad_pj[i][alpha] - grad_mj[i][alpha] + grad_pi[j][beta] - grad_mi[j][beta] );

                    // Un-Displace
                    spins_pi[i][alpha] -= delta_;
                    spins_mi[i][alpha] += delta_;
                    spins_pj[j][beta] -= delta_;
                    spins_mj[j][beta] += delta_;
                }
            }
        }
    }
}

void Hamiltonian::Gradient_and_Energy( const vectorfield & spins, vectorfield & gradient, scalar & energy )
{
    Gradient( spins, gradient );
    energy = Energy( spins );
}

void Hamiltonian::Gradient_FD( const vectorfield & spins, vectorfield & gradient )
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
            spins_plus[i][dim] += delta_;
            spins_minus[i][dim] -= delta_;

            // Calculate gradient component
            scalar E_plus    = Energy( spins_plus );
            scalar E_minus   = Energy( spins_minus );
            gradient[i][dim] = 0.5 * ( E_plus - E_minus ) / delta_;

            // Un-Displace
            spins_plus[i][dim] -= delta_;
            spins_minus[i][dim] += delta_;
        }
    }
}

scalar Hamiltonian::Energy( const vectorfield & spins )
{
    scalar sum  = 0;
    auto energy = Energy_Contributions( spins );
    for( const auto & E : energy )
        sum += E.second;
    return sum;
}

std::vector<std::pair<std::string, scalar>> Hamiltonian::Energy_Contributions( const vectorfield & spins )
{
    Energy_Contributions_per_Spin( spins, energy_contributions_per_spin_ );
    std::vector<std::pair<std::string, scalar>> energy( energy_contributions_per_spin_.size() );
    for( std::size_t i = 0; i < energy.size(); ++i )
    {
        energy[i]
            = { energy_contributions_per_spin_[i].first, Vectormath::sum( energy_contributions_per_spin_[i].second ) };
    }
    return energy;
}

std::size_t Hamiltonian::Number_of_Interactions() const
{
    return energy_contributions_per_spin_.size();
}

} // namespace Engine
