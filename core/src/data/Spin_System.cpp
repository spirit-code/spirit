#include <data/Spin_System.hpp>
#include <engine/Neighbours.hpp>
#include <engine/Vectormath.hpp>
#include <io/IO.hpp>

#include <numeric>
#include <iostream>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <random>

namespace Data
{
    Spin_System::Spin_System(std::unique_ptr<Engine::Hamiltonian> hamiltonian, 
        std::shared_ptr<Geometry> geometry,
        std::unique_ptr<Parameters_Method_LLG> llg_params, 
        std::unique_ptr<Parameters_Method_MC>  mc_params, 
        std::unique_ptr<Parameters_Method_EMA> ema_params,
        std::unique_ptr<Parameters_Method_MMF> mmf_params,
        bool iteration_allowed) :
        iteration_allowed(iteration_allowed), hamiltonian(std::move(hamiltonian)), geometry(geometry),
        llg_parameters(std::move(llg_params)), mc_parameters(std::move(mc_params)),
        ema_parameters(std::move(ema_params)), mmf_parameters(std::move(mmf_params))
    {
        // Get Number of Spins
        this->nos = this->geometry->nos;

        // Initialize Spins Array
        this->spins = std::shared_ptr<vectorfield>(new vectorfield(nos));
        
        // Initialize Modes container
        this->modes = std::vector<std::shared_ptr<vectorfield>>(this->ema_parameters->n_modes, NULL);

        // Initialize Eigenvalues vector
        this->eigenvalues = std::vector<scalar>(this->modes.size(),0);

        // ...
        this->E = 0;
        this->E_array = std::vector<std::pair<std::string, scalar>>(0);
        this->M = Vector3{0,0,0};
        this->effective_field = vectorfield(this->nos);

    }//end Spin_System constructor


    // Copy Constructor
    Spin_System::Spin_System(Spin_System const & other)
    {
        this->nos = other.nos;
        this->spins = std::shared_ptr<vectorfield>(new vectorfield(*other.spins));
        this->modes = std::vector<std::shared_ptr<vectorfield>>(other.modes.size(), NULL);
        this->eigenvalues = other.eigenvalues; 
        
        // copy the modes
        for (int i=0; i<other.modes.size(); i++)
            if ( other.modes[i] != NULL ) 
                this->modes[i] = 
                    std::shared_ptr<vectorfield>(new vectorfield(*other.modes[i]));

        this->E = other.E;
        this->E_array = other.E_array;
        this->effective_field = other.effective_field;

        this->geometry = std::shared_ptr<Data::Geometry>(new Data::Geometry(*other.geometry));

        if (other.hamiltonian->Name() == "Heisenberg")
        {
            this->hamiltonian = std::shared_ptr<Engine::Hamiltonian>(new Engine::Hamiltonian_Heisenberg(*(Engine::Hamiltonian_Heisenberg*)(other.hamiltonian.get())));
        }
        else if (other.hamiltonian->Name() == "Gaussian")
        {
            this->hamiltonian = std::shared_ptr<Engine::Hamiltonian>(new Engine::Hamiltonian_Gaussian(*(Engine::Hamiltonian_Gaussian*)(other.hamiltonian.get())));
        }

        this->llg_parameters = std::shared_ptr<Data::Parameters_Method_LLG>(new Data::Parameters_Method_LLG(*other.llg_parameters));

        this->mc_parameters = std::shared_ptr<Data::Parameters_Method_MC>(new Data::Parameters_Method_MC(*other.mc_parameters));

        this->ema_parameters = std::shared_ptr<Data::Parameters_Method_EMA>(new Data::Parameters_Method_EMA(*other.ema_parameters));

        this->mmf_parameters = std::shared_ptr<Data::Parameters_Method_MMF>(new Data::Parameters_Method_MMF(*other.mmf_parameters));

        this->iteration_allowed = false;
    }


    // Copy Assignment operator
    Spin_System& Spin_System::operator=(Spin_System const & other)
    {
        if (this != &other)
        {
            this->nos = other.nos;
            this->spins = std::shared_ptr<vectorfield>(new vectorfield(*other.spins));
            this->modes = std::vector<std::shared_ptr<vectorfield>>(other.modes.size(), NULL);
            this->eigenvalues = other.eigenvalues; 
            
            // copy the modes
            for (int i=0; i<other.modes.size(); i++)
                if ( other.modes[i] != NULL ) 
                    this->modes[i] = 
                        std::shared_ptr<vectorfield>(new vectorfield(*other.modes[i]));

            this->E = other.E;
            this->E_array = other.E_array;
            this->effective_field = other.effective_field;

            this->geometry = std::shared_ptr<Data::Geometry>(new Data::Geometry(*other.geometry));

            if (other.hamiltonian->Name() == "Heisenberg")
            {
                this->hamiltonian = std::shared_ptr<Engine::Hamiltonian>(new Engine::Hamiltonian_Heisenberg(*(Engine::Hamiltonian_Heisenberg*)(other.hamiltonian.get())));
            }
            else if (other.hamiltonian->Name() == "Gaussian")
            {
                this->hamiltonian = std::shared_ptr<Engine::Hamiltonian>(new Engine::Hamiltonian_Gaussian(*(Engine::Hamiltonian_Gaussian*)(other.hamiltonian.get())));
            }

            this->llg_parameters = std::shared_ptr<Data::Parameters_Method_LLG>(new Data::Parameters_Method_LLG(*other.llg_parameters));

            this->mc_parameters = std::shared_ptr<Data::Parameters_Method_MC>(new Data::Parameters_Method_MC(*other.mc_parameters));

            this->ema_parameters = std::shared_ptr<Data::Parameters_Method_EMA>(new Data::Parameters_Method_EMA(*other.ema_parameters));

            this->iteration_allowed = false;
        }

        return *this;
    }


    void Spin_System::UpdateEnergy()
    {
        this->E_array = this->hamiltonian->Energy_Contributions(*this->spins);
        scalar sum = 0;
        for (auto E : E_array) sum += E.second;
        this->E = sum;
    }


    void Spin_System::UpdateEffectiveField()
    {
        this->hamiltonian->Gradient(*this->spins, this->effective_field);
        Engine::Vectormath::scale(this->effective_field, -1);
    }


    void Spin_System::Lock() const
    {
        try
        {
            this->mutex.lock();
        }
        catch( ... )
        {
            spirit_handle_exception_core("Locking the Spin_System failed!");
        }
    }


    void Spin_System::Unlock() const
    {
        try
        {
            this->mutex.unlock();
        }
        catch( ... )
        {
            spirit_handle_exception_core("Unlocking the Spin_System failed!");
        }
    }
}