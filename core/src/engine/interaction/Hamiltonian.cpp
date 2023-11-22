#include <data/Spin_System.hpp>
#include <engine/Backend_par.hpp>
#include <engine/Hamiltonian.hpp>
#include <engine/Indexing.hpp>
#include <engine/Neighbours.hpp>
#include <engine/Vectormath.hpp>
#include <utility/Constants.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>

using namespace Data;
using namespace Utility;

using Engine::Indexing::check_atom_type;

namespace Engine
{

Hamiltonian::Hamiltonian( std::shared_ptr<Geometry> geometry, intfield boundary_conditions )
        : geometry( std::move( geometry ) ),
          boundary_conditions( std::move( boundary_conditions ) ),
          name_update_paused( false ),
          hamiltonian_class( HAMILTONIAN_CLASS::GENERIC )
{
    prng             = std::mt19937( 94199188 );
    distribution_int = std::uniform_int_distribution<int>( 0, 1 );
    this->updateInteractions(); // should be a nop, but has to be here semantically
    this->updateName();
}

Hamiltonian::Hamiltonian( const Hamiltonian & other )
        : geometry( other.geometry ),
          boundary_conditions( other.boundary_conditions ),
          interactions( 0 ),
          active_interactions_size( other.active_interactions_size ),
          common_interactions_size( other.common_interactions_size ),
          prng( other.prng ),
          distribution_int( other.distribution_int ),
          name_update_paused( other.name_update_paused ),
          hamiltonian_class( other.hamiltonian_class ),
          class_name( other.class_name )
{
    interactions.reserve( other.interactions.capacity() );
    for( const auto & interaction : other.interactions )
        interactions.emplace_back( interaction->clone( this ) );
}

void Hamiltonian::updateActiveInteractions()
{
    // take inventory and put the interactions that contribute to the front of the vector
    const auto is_active                 = []( const auto & i ) { return i->is_active(); };
    const auto active_partition_boundary = std::partition( begin( interactions ), end( interactions ), is_active );
    active_interactions_size             = std::distance( begin( interactions ), active_partition_boundary );

    // sort by spin order (may speed up predictions)
    const auto has_common_spin_order = []( const auto & i ) { return i->spin_order() == common_spin_order; };
    const auto common_partition_boundary
        = std::partition( begin( interactions ), active_partition_boundary, has_common_spin_order );
    common_interactions_size = std::distance( begin( interactions ), common_partition_boundary );
}

void Hamiltonian::updateInteractions()
{
    for( const auto & interaction : interactions )
    {
        interaction->updateGeometry();
    }

    // Update, which terms still contribute
    this->updateActiveInteractions();
}

void Hamiltonian::Energy_per_Spin( const vectorfield & spins, scalarfield & energy_per_spin )
{
    const auto nos = spins.size();
    // Allocate if not already allocated
    if( energy_per_spin.size() != nos )
        energy_per_spin = scalarfield( nos, 0 );
    // Otherwise set to zero
    else
        Vectormath::fill( energy_per_spin, 0 );

    for( const auto & interaction : getActiveInteractions() )
    {
        interaction->Energy_per_Spin( spins, energy_per_spin );
    }
}

void Hamiltonian::Energy_Contributions_per_Spin( const vectorfield & spins, vectorlabeled<scalarfield> & contributions )
{
    const auto nos = spins.size();

    if( contributions.size() != active_interactions_size )
    {
        contributions = std::vector( active_interactions_size, std::pair{ std::string_view{}, scalarfield( nos, 0 ) } );
    }
    else
    {
        for( auto & contrib : contributions )
        {
            // Allocate if not already allocated
            if( contrib.second.size() != nos )
                contrib.second = scalarfield( nos, 0 );
            // Otherwise set to zero
            else
                Vectormath::fill( contrib.second, 0 );
        }
    }

    const auto & active_interactions = getActiveInteractions();
    for( std::size_t i = 0; i < active_interactions.size(); ++i )
    {
        contributions[i].first = active_interactions[i]->Name();
        active_interactions[i]->Energy_per_Spin( spins, contributions[i].second );
    }
}

scalar Hamiltonian::Energy_Single_Spin( int ispin, const vectorfield & spins )
{
    scalar energy = 0;
    if( check_atom_type( this->geometry->atom_types[ispin] ) )
    {
        for( const auto & interaction : getActiveInteractions() )
        {
            energy += interaction->Energy_Single_Spin( ispin, spins );
        }
    }
    return energy;
}

void Hamiltonian::Gradient( const vectorfield & spins, vectorfield & gradient )
{
    // Set to zero
    Vectormath::fill( gradient, { 0, 0, 0 } );

    for( const auto & interaction : getActiveInteractions() )
    {
        interaction->Gradient( spins, gradient );
    }
}

void Hamiltonian::Gradient_and_Energy( const vectorfield & spins, vectorfield & gradient, scalar & energy )
{
    // Set to zero
    Vectormath::fill( gradient, { 0, 0, 0 } );
    energy = 0;

    const auto N              = spins.size();
    const auto * s            = spins.data();
    const auto * g            = gradient.data();
    static constexpr scalar c = 1.0 / static_cast<scalar>( common_spin_order );

    for( const auto & interaction : getCommonInteractions() )
    {
        interaction->Gradient( spins, gradient );
    }

    energy += Backend::par::reduce( N, [s, g] SPIRIT_LAMBDA( int idx ) { return c * g[idx].dot( s[idx] ); } );

    for( const auto & interaction : getUncommonInteractions() )
    {
        interaction->Gradient( spins, gradient );
        energy += interaction->Energy( spins );
    }
}

void Hamiltonian::Hessian( const vectorfield & spins, MatrixX & hessian )
{
    // --- Set to zero
    hessian.setZero();

    for( const auto & interaction : getActiveInteractions() )
    {
        interaction->Hessian( spins, hessian );
    }
}

void Hamiltonian::Sparse_Hessian( const vectorfield & spins, SpMatrixX & hessian )
{
    std::size_t sparse_size_per_cell = 0;
    for( const auto & interaction : getActiveInteractions() )
        sparse_size_per_cell += interaction->Sparse_Hessian_Size_per_Cell();

    std::vector<Interaction::triplet> tripletList;
    tripletList.reserve( geometry->n_cells_total * sparse_size_per_cell );

    for( const auto & interaction : getActiveInteractions() )
    {
        interaction->Sparse_Hessian( spins, tripletList );
    }

    hessian.setFromTriplets( tripletList.begin(), tripletList.end() );
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
    scalar sum = 0;
    for( const auto & interaction : getActiveInteractions() )
    {
        sum += interaction->Energy( spins );
    }
    return sum;
}

Data::vectorlabeled<scalar> Hamiltonian::Energy_Contributions( const vectorfield & spins )
{
    vectorlabeled<scalar> contributions( 0 );
    contributions.reserve( getActiveInteractionsSize() );
    for( const auto & interaction : getActiveInteractions() )
    {
        contributions.emplace_back( interaction->Name(), interaction->Energy( spins ) );
    }
    return contributions;
}

void Hamiltonian::updateName()
{
    if( name_update_paused )
        return;

    if( interactions.size() == 1 && hasInteraction<Interaction::Gaussian>() )
        hamiltonian_class = HAMILTONIAN_CLASS::GAUSSIAN;
    else if( !hasInteraction<Interaction::Gaussian>() )
        hamiltonian_class = HAMILTONIAN_CLASS::HEISENBERG;
    else
        hamiltonian_class = HAMILTONIAN_CLASS::GENERIC;

    class_name = hamiltonianClassName( hamiltonian_class );
}

// Hamiltonian name as string
std::string_view Hamiltonian::Name() const
{
    return class_name;
};

} // namespace Engine
