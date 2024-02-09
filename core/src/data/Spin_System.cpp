#include <data/Spin_System.hpp>
#include <engine/Hamiltonian_Defines.hpp>
#include <engine/Neighbours.hpp>
#include <engine/Vectormath.hpp>
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
    std::unique_ptr<Hamiltonian> hamiltonian, std::shared_ptr<Geometry> geometry,
    std::unique_ptr<Parameters_Method_LLG> llg_params, std::unique_ptr<Parameters_Method_MC> mc_params,
    std::unique_ptr<Parameters_Method_EMA> ema_params, std::unique_ptr<Parameters_Method_MMF> mmf_params,
    bool iteration_allowed )
try : iteration_allowed( iteration_allowed ), singleshot_allowed( false ), hamiltonian( std::move( hamiltonian ) ),
    geometry( geometry ), llg_parameters( std::move( llg_params ) ), mc_parameters( std::move( mc_params ) ),
    ema_parameters( std::move( ema_params ) ), mmf_parameters( std::move( mmf_params ) )
{
    // Get Number of Spins
    this->nos = this->geometry->nos;

    // Initialize Spins Array
    this->spins = std::make_shared<vectorfield>( nos );

    // Initialize Modes container
    this->modes = std::vector<std::shared_ptr<vectorfield>>( this->ema_parameters->n_modes, NULL );

    // Initialize Eigenvalues vector
    this->eigenvalues = std::vector<scalar>( this->modes.size(), 0 );

    // ...
    this->E               = 0;
    this->E_array         = vectorlabeled<scalar>( 0 );
    this->M               = Vector3{ 0, 0, 0 };
    this->effective_field = vectorfield( this->nos );
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
    this->spins       = std::make_unique<vectorfield>( *other.spins );
    this->modes       = std::vector<std::shared_ptr<vectorfield>>( other.modes.size(), NULL );
    this->eigenvalues = other.eigenvalues;

    // copy the modes
    for( int i = 0; i < other.modes.size(); i++ )
        if( other.modes[i] != NULL )
            this->modes[i] = std::make_shared<vectorfield>( *other.modes[i] );

    this->E               = other.E;
    this->E_array         = other.E_array;
    this->effective_field = other.effective_field;

    this->geometry    = std::make_shared<Data::Geometry>( *other.geometry );
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
        this->spins       = std::make_unique<vectorfield>( *other.spins );
        this->modes       = std::vector<std::shared_ptr<vectorfield>>( other.modes.size(), NULL );
        this->eigenvalues = other.eigenvalues;

        // copy the modes
        for( int i = 0; i < other.modes.size(); i++ )
            if( other.modes[i] != NULL )
                this->modes[i] = std::make_shared<vectorfield>( *other.modes[i] );

        this->E               = other.E;
        this->E_array         = other.E_array;
        this->effective_field = other.effective_field;

        this->geometry    = std::make_shared<Data::Geometry>( *other.geometry );
        this->hamiltonian = std::make_shared<Engine::Hamiltonian>( *other.hamiltonian );

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
    this->E_array = this->hamiltonian->Energy_Contributions( *this->spins );
    scalar sum    = 0;
    for( auto & E_item : E_array )
        sum += E_item.second;
    this->E = sum;
}
catch( ... )
{
    spirit_rethrow( "Spin_System::UpdateEnergy failed" );
}

template<typename Hamiltonian>
void Spin_System<Hamiltonian>::UpdateEffectiveField()
try
{
    this->hamiltonian->Gradient( *this->spins, this->effective_field );
    Engine::Vectormath::scale( this->effective_field, -1 );
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

template class Data::Spin_System<Engine::Hamiltonian>;
