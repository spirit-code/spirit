#include <data/Spin_System_Chain.hpp>

namespace Data
{
	Spin_System_Chain::Spin_System_Chain(std::vector<std::shared_ptr<Spin_System>> images, std::shared_ptr<Data::Parameters_Method_GNEB> gneb_parameters, bool iteration_allowed) :
		gneb_parameters(gneb_parameters)
	{
		this->noi = images.size();
		this->images = images;
		//this->gneb_parameters = gneb_parameters;

		this->iteration_allowed = iteration_allowed;
		this->idx_active_image = 0;

		this->image_type = std::vector<GNEB_Image_Type>(this->noi, GNEB_Image_Type::Normal);

		this->Rx = std::vector<scalar>(this->noi, 0);
		this->Rx_interpolated = std::vector<scalar>(this->noi + (this->noi - 1)*gneb_parameters->n_E_interpolations, 0);
		this->E_interpolated = std::vector<scalar>(this->noi + (this->noi - 1)*gneb_parameters->n_E_interpolations, 0);
		this->E_array_interpolated = std::vector<std::vector<scalar>>(7, std::vector<scalar>(this->noi + (this->noi-1)*gneb_parameters->n_E_interpolations, 0));
	}

	void Spin_System_Chain::Lock() const
	{
		this->mutex.lock();
		for (auto& image : this->images) image->Lock();
	}

	void Spin_System_Chain::Unlock() const
	{
		this->mutex.unlock();
		for (auto& image : this->images) image->Unlock();
	}
}