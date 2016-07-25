#include "Spin_System_Chain.h"
#include "Manifoldmath.h"

namespace Data
{
	Spin_System_Chain::Spin_System_Chain(std::vector<std::shared_ptr<Spin_System>> images, std::shared_ptr<Data::Parameters_GNEB> gneb_parameters, bool iteration_allowed) :
		gneb_parameters(gneb_parameters)
	{
		this->noi = images.size();
		this->images = images;
		//this->gneb_parameters = gneb_parameters;

		this->iteration_allowed = iteration_allowed;
		this->idx_active_image = 0;

		this->climbing_image = std::vector<bool>(this->noi, false);
		this->falling_image = std::vector<bool>(this->noi, false);

		this->Setup_Initial_Data();
	}

	void Spin_System_Chain::Setup_Initial_Data()
	{
		this->Rx = std::vector<double>(this->noi, 0);
		this->Rx_interpolated = std::vector<double>((this->noi - 1)*gneb_parameters->n_E_interpolations, 0);
		this->E_interpolated = std::vector<double>((this->noi - 1)*gneb_parameters->n_E_interpolations, 0);
		this->E_array_interpolated = std::vector<std::vector<double>>(7, std::vector<double>((this->noi - 1)*gneb_parameters->n_E_interpolations, 0));

		this->tangents = std::vector<std::vector<double>>(noi, std::vector<double>(3 * this->images[0]->nos));

		// Initial data update
		this->Update_Data();
	}


	void Spin_System_Chain::Update_Data()
	{
		for (int i = 0; i < this->noi; ++i)
		{
			//Engine::Energy::Update(*c->images[i]);
			//c->images[i]->E = c->images[i]->hamiltonian_isotropic->Energy(c->images[i]->spins);
			this->images[i]->UpdateEnergy();
			if (i > 0) this->Rx[i] = this->Rx[i - 1] + Utility::Manifoldmath::Dist_Geodesic(this->images[i - 1]->spins, this->images[i]->spins);
		}
	}


	void Spin_System_Chain::Insert_Image_Before(int idx, std::shared_ptr<Data::Spin_System> system)
	{
		this->noi++;
		this->images.insert(this->images.begin() + idx, system);
		
		this->climbing_image.insert(this->climbing_image.begin() + idx, false);
		this->falling_image.insert(this->falling_image.begin() + idx, false);

		this->Setup_Initial_Data();
	}

	void Spin_System_Chain::Insert_Image_After(int idx, std::shared_ptr<Data::Spin_System> system)
	{
		if (idx < noi - 1) this->Insert_Image_Before(idx + 1, system);
		else
		{
			this->noi++;
			this->images.push_back(system);

			this->climbing_image.push_back(false);
			this->falling_image.push_back(false);

			this->Setup_Initial_Data();
		}
	}

	void Spin_System_Chain::Replace_Image(int idx, std::shared_ptr<Data::Spin_System> system)
	{
		this->images[idx] = system;

		this->Update_Data();
	}

	void Spin_System_Chain::Delete_Image(int idx)
	{
		this->noi--;
		this->images.erase(this->images.begin() + idx);
		
		this->climbing_image.erase(this->climbing_image.begin() + idx);
		this->falling_image.erase(this->falling_image.begin() + idx);

		this->Setup_Initial_Data();
	}
}