#include <data/Spin_System.hpp>
#include <engine/Neighbours.hpp>
#include <engine/Vectormath.hpp>
#include <engine/spin/Hamiltonian.hpp>
#include <io/IO.hpp>

#include <algorithm>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>

namespace Data
{

template<typename Hamiltonian>
Spin_System<Hamiltonian>::Spin_System(
    std::unique_ptr<Hamiltonian> hamiltonian, std::unique_ptr<Parameters_Method_LLG> llg_params,
    std::unique_ptr<Parameters_Method_MC> mc_params, std::unique_ptr<Parameters_Method_EMA> ema_params,
    std::unique_ptr<Parameters_Method_MMF> mmf_params, bool iteration_allowed )
try : hamiltonian( std::move( hamiltonian ) ), llg_parameters( std::move( llg_params ) ),
    mc_parameters( std::move( mc_params ) ), ema_parameters( std::move( ema_params ) ),
    mmf_parameters( std::move( mmf_params ) ), iteration_allowed( iteration_allowed ), singleshot_allowed( false )
{
    // Get Number of Spins
    this->nos = this->hamiltonian->get_geometry().nos;

    // Initialize Spins Array
    this->state = std::make_shared<typename Hamiltonian::state_t>( Engine::make_state<typename Hamiltonian::state_t>( nos ) );

    // Initialize Modes container
    this->modes = std::vector<std::optional<vectorfield>>( this->ema_parameters->n_modes, std::nullopt );

    // Initialize Eigenvalues vector
    this->eigenvalues = std::vector<scalar>( this->modes.size(), 0 );

    // ...
    this->E = System_Energy{ 0, vectorlabeled<scalar>( 0 ), vectorlabeled<scalarfield>( 0 ) };
    this->M = System_Magnetization{ Vector3{ 0, 0, 0 }, vectorfield( this->nos ) };
}
catch( ... )
{
    spirit_rethrow( "Spin system initialisation failed" );
}

// Copy Constructor
template<typename Hamiltonian>
Spin_System<Hamiltonian>::Spin_System( Spin_System const & other )
try
{
    this->nos         = other.nos;
    this->state       = std::make_shared<typename Hamiltonian::state_t>( *other.state );
    this->modes       = other.modes;
    this->eigenvalues = other.eigenvalues;

    this->E = other.E;
    this->M = other.M;

    this->hamiltonian = std::make_shared<Hamiltonian>( *other.hamiltonian );

    this->llg_parameters = std::make_shared<Data::Parameters_Method_LLG>( *other.llg_parameters );
    this->mc_parameters  = std::make_shared<Data::Parameters_Method_MC>( *other.mc_parameters );
    this->ema_parameters = std::make_shared<Data::Parameters_Method_EMA>( *other.ema_parameters );
    this->mmf_parameters = std::make_shared<Data::Parameters_Method_MMF>( *other.mmf_parameters );

    this->iteration_allowed = false;
}
catch( ... )
{
    spirit_rethrow( "Copy-assigning spin system failed" );
}

// Copy assignment operator
template<typename Hamiltonian>
Spin_System<Hamiltonian> & Spin_System<Hamiltonian>::operator=( Spin_System<Hamiltonian> const & other )
try
{
    if( this != &other )
    {
        this->nos         = other.nos;
        this->state       = std::make_unique<typename Hamiltonian::state_t>( *other.state );
        this->modes       = other.modes;
        this->eigenvalues = other.eigenvalues;

        this->E = other.E;
        this->M = other.M;

        this->hamiltonian = std::make_shared<Hamiltonian>( *other.hamiltonian );

        this->llg_parameters = std::make_shared<Data::Parameters_Method_LLG>( *other.llg_parameters );
        this->mc_parameters  = std::make_shared<Data::Parameters_Method_MC>( *other.mc_parameters );
        this->ema_parameters = std::make_shared<Data::Parameters_Method_EMA>( *other.ema_parameters );
        this->mmf_parameters = std::make_shared<Data::Parameters_Method_MMF>( *other.mmf_parameters );

        this->iteration_allowed = false;
    }

    return *this;
}
catch( ... )
{
    spirit_rethrow( "Copy-assigning spin system failed" );
    return *this;
}

template<typename Hamiltonian>
void Spin_System<Hamiltonian>::UpdateEnergy()
try
{
    this->E.per_interaction = this->hamiltonian->Energy_Contributions( *this->state );
    scalar sum              = 0;
    for( auto & E_item : E.per_interaction )
        sum += E_item.second;
    this->E.total = sum;
}
catch( ... )
{
    spirit_rethrow( "Spin_System::UpdateEnergy failed" );
}

template<typename Hamiltonian>
void Spin_System<Hamiltonian>::UpdateEffectiveField()
try
{
    this->hamiltonian->Gradient( *this->state, this->M.effective_field );
    Engine::Vectormath::scale( this->M.effective_field, -1 );
}
catch( ... )
{
    spirit_rethrow( "Spin_System::UpdateEffectiveField failed" );
}

template<typename Hamiltonian>
void Spin_System<Hamiltonian>::Lock() noexcept
try
{
    this->ordered_lock.lock();
}
catch( ... )
{
    spirit_handle_exception_core( "Locking the Spin_System failed!" );
}

template<typename Hamiltonian>
void Spin_System<Hamiltonian>::Unlock() noexcept
try
{
    this->ordered_lock.unlock();
}
catch( ... )
{
    spirit_handle_exception_core( "Unlocking the Spin_System failed!" );
}

} // namespace Data

template class Data::Spin_System<Engine::Spin::HamiltonianVariant>;
