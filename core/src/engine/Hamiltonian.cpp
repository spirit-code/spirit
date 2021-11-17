#include <engine/Hamiltonian.hpp>
#include <engine/Vectormath.hpp>
#include <utility/Exception.hpp>
#include <utility/Logging.hpp>

using namespace Utility;

namespace Engine
{

Hamiltonian::Hamiltonian( intfield boundary_conditions ) : boundary_conditions( boundary_conditions )
{
    prng             = std::mt19937( 94199188 );
    distribution_int = std::uniform_int_distribution<int>( 0, 1 );

    delta = 1e-3;
}

void Hamiltonian::Update_Energy_Contributions()
{
    // Not Implemented!
    spirit_throw(
        Exception_Classifier::Not_Implemented, Log_Level::Error,
        "Tried to use  Hamiltonian::Update_Energy_Contributions() of the Hamiltonian base class!" );
}

void Hamiltonian::Hessian( const vectorfield & spins, MatrixX & hessian )
{
    this->Hessian_FD( spins, hessian );
}

void Hamiltonian::Sparse_Hessian( const vectorfield & spins, SpMatrixX & hessian )
{
    // Not implemented
    return;
}

void Hamiltonian::Hessian_FD( const vectorfield & spins, MatrixX & hessian )
{
    // This is a regular finite difference implementation (probably not very efficient)
    // using the differences between gradient values (not function)
    // see https://v8doc.sas.com/sashtml/ormp/chap5/sect28.htm

    int nos = spins.size();

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

    for( int i = 0; i < nos; ++i )
    {
        for( int j = 0; j < nos; ++j )
        {
            for( int alpha = 0; alpha < 3; ++alpha )
            {
                for( int beta = 0; beta < 3; ++beta )
                {
                    // Displace
                    spins_pi[i][alpha] += delta;
                    spins_mi[i][alpha] -= delta;
                    spins_pj[j][beta] += delta;
                    spins_mj[j][beta] -= delta;

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

void Hamiltonian::Gradient( const vectorfield & spins, vectorfield & gradient )
{
    this->Gradient_FD( spins, gradient );
}

void Hamiltonian::Gradient_and_Energy( const vectorfield & spins, vectorfield & gradient, scalar & energy )
{
    this->Gradient( spins, gradient );
    energy = this->Energy( spins );
}

void Hamiltonian::Gradient_FD( const vectorfield & spins, vectorfield & gradient )
{
    int nos = spins.size();

    // Calculate finite difference
    vectorfield spins_plus( nos );
    vectorfield spins_minus( nos );

    spins_plus  = spins;
    spins_minus = spins;

    for( int i = 0; i < nos; ++i )
    {
        for( int dim = 0; dim < 3; ++dim )
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

scalar Hamiltonian::Energy( const vectorfield & spins )
{
    scalar sum  = 0;
    auto energy = Energy_Contributions( spins );
    for( auto E : energy )
        sum += E.second;
    return sum;
}

std::vector<std::pair<std::string, scalar>> Hamiltonian::Energy_Contributions( const vectorfield & spins )
{
    Energy_Contributions_per_Spin( spins, this->energy_contributions_per_spin );
    std::vector<std::pair<std::string, scalar>> energy( this->energy_contributions_per_spin.size() );
    for( unsigned int i = 0; i < energy.size(); ++i )
    {
        energy[i] = { this->energy_contributions_per_spin[i].first,
                      Vectormath::sum( this->energy_contributions_per_spin[i].second ) };
    }
    return energy;
}

void Hamiltonian::Energy_Contributions_per_Spin(
    const vectorfield & spins, std::vector<std::pair<std::string, scalarfield>> & contributions )
{
    // Not Implemented!
    spirit_throw(
        Exception_Classifier::Not_Implemented, Log_Level::Error,
        "Tried to use  Hamiltonian::Energy_Contributions_per_Spin() of the Hamiltonian base class!" );
}

int Hamiltonian::Number_of_Interactions()
{
    return energy_contributions_per_spin.size();
}

scalar Hamiltonian::Energy_Single_Spin( int ispin, const vectorfield & spins )
{
    // Not Implemented!
    spirit_throw(
        Exception_Classifier::Not_Implemented, Log_Level::Error,
        "Tried to use  Hamiltonian::Energy_Single_Spin() of the Hamiltonian base class!" );
}

static const std::string name = "--";
const std::string & Hamiltonian::Name()
{
    spirit_throw(
        Exception_Classifier::Not_Implemented, Log_Level::Error,
        "Tried to use  Hamiltonian::Name() of the Hamiltonian base class!" );
}

} // namespace Engine