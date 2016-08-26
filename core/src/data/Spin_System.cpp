#include "Spin_System.h"
#include "Neighbours.h"
#include "Vectormath.h"
#include "Vectoroperators.h"
#include "IO.h"

#include <numeric>
#include <iostream>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <random>

using namespace Utility;

namespace Data
{
	Spin_System::Spin_System(std::unique_ptr<Engine::Hamiltonian> hamiltonian, std::unique_ptr<Geometry> geometry, std::unique_ptr<Parameters_Method_LLG> llg_params, bool iteration_allowed) :
		iteration_allowed(iteration_allowed), hamiltonian(std::move(hamiltonian)), geometry(std::move(geometry)), llg_parameters(std::move(llg_params))
	{

		// Get Number of Spins
		this->nos = this->geometry->nos;

		// Initialize Spins Array
		this->spins = std::shared_ptr<std::vector<double>>(new std::vector<double>(3 * nos));

		// Check for the type of the Hamiltonian
		if (typeid(*this->hamiltonian.get()) == typeid(Engine::Hamiltonian_Isotropic))
		{
			is_isotropic = true;
		}
		else if (typeid(*this->hamiltonian.get()) == typeid(Engine::Hamiltonian_Anisotropic))
		{
			is_isotropic = false;
		}

		// ...
		this->E = 0;
		this->E_array = std::vector<double>(7, 0.0);
		this->effective_field = std::vector<double>(3 * this->nos, 0);

	}//end Spin_System constructor

	 // Copy Constructor
	Spin_System::Spin_System(Spin_System const & other)
	{
		this->nos = other.nos;
		this->spins = std::shared_ptr<std::vector<double>>(new std::vector<double>(*other.spins));

		this->is_isotropic = other.is_isotropic;

		this->E = other.E;
		this->E_array = other.E_array;
		this->effective_field = other.effective_field;

		this->geometry = std::shared_ptr<Data::Geometry>(new Data::Geometry(*other.geometry));
		
		// Getting the Hamiltonian involves UGLY casting... maybe there's a nicer (and safer) way?
		if (this->is_isotropic)
		{
			this->hamiltonian = std::shared_ptr<Engine::Hamiltonian>(new Engine::Hamiltonian_Isotropic(*(Engine::Hamiltonian_Isotropic*)(other.hamiltonian.get())));
		}
		else
		{
			this->hamiltonian = std::shared_ptr<Engine::Hamiltonian>(new Engine::Hamiltonian_Anisotropic(*(Engine::Hamiltonian_Anisotropic*)(other.hamiltonian.get())));
		}

		this->llg_parameters = std::shared_ptr<Data::Parameters_Method_LLG>(new Data::Parameters_Method_LLG(*other.llg_parameters));

		this->iteration_allowed = false;
	}

	// Assignment operator
	Spin_System& Spin_System::operator=(Spin_System const & other)
	{
		if (this != &other)
		{
			this->nos = other.nos;
			this->spins = std::shared_ptr<std::vector<double>>(new std::vector<double>(*other.spins));

			this->is_isotropic = other.is_isotropic;

			this->E = other.E;
			this->E_array = other.E_array;
			this->effective_field = other.effective_field;

			this->geometry = std::shared_ptr<Data::Geometry>(new Data::Geometry(*other.geometry));
			
			// Getting the Hamiltonian involves UGLY casting... maybe there's a nicer (and safer) way?
			if (this->is_isotropic)
			{
				this->hamiltonian = std::shared_ptr<Engine::Hamiltonian>(new Engine::Hamiltonian_Isotropic(*(Engine::Hamiltonian_Isotropic*)(other.hamiltonian.get())));
			}
			else
			{
				this->hamiltonian = std::shared_ptr<Engine::Hamiltonian>(new Engine::Hamiltonian_Anisotropic(*(Engine::Hamiltonian_Anisotropic*)(other.hamiltonian.get())));
			}

			this->llg_parameters = std::shared_ptr<Data::Parameters_Method_LLG>(new Data::Parameters_Method_LLG(*other.llg_parameters));

			this->iteration_allowed = false;
		}
		return *this;

	}

	void Spin_System::UpdateEnergy()
	{
		this->E_array = this->hamiltonian->Energy_Array(*this->spins);
		this->E = sum(this->E_array);
	}

	void Spin_System::UpdateEffectiveField()
	{
		this->hamiltonian->Effective_Field(*this->spins, this->effective_field);
	}

}