#include <data/Spin_System_Chain.hpp>
#include <utility/Exception.hpp>

namespace Data
{
    Spin_System_Chain::Spin_System_Chain(std::vector<std::shared_ptr<Spin_System>> images, std::shared_ptr<Data::Parameters_Method_GNEB> gneb_parameters, bool iteration_allowed) :
        iteration_allowed(iteration_allowed), singleshot_allowed(false),
        gneb_parameters(gneb_parameters)
    {
        this->noi = images.size();
        this->images = images;
        //this->gneb_parameters = gneb_parameters;

        this->idx_active_image = 0;

        this->image_type = std::vector<GNEB_Image_Type>(this->noi, GNEB_Image_Type::Normal);

        this->Rx = std::vector<scalar>(this->noi, 0);
        int size_interpolated = this->noi + (this->noi - 1)*gneb_parameters->n_E_interpolations;
        this->Rx_interpolated = std::vector<scalar>(size_interpolated, 0);
        this->E_interpolated = std::vector<scalar>(size_interpolated, 0);
        this->E_array_interpolated = std::vector<std::vector<scalar>>(7, std::vector<scalar>(size_interpolated, 0));

        // Zero the HTST info
        this->htst_info.eigenvalues_min        = scalarfield(0);
        this->htst_info.eigenvalues_sp         = scalarfield(0);
        this->htst_info.perpendicular_velocity = scalarfield(0);

        this->htst_info.temperature_exponent = 0;
        this->htst_info.me = 0;
        this->htst_info.Omega_0 = 0;
        this->htst_info.s = 0;
        this->htst_info.volume_min = 0;
        this->htst_info.volume_sp = 0;
        this->htst_info.prefactor_dynamical = 0;
        this->htst_info.prefactor = 0;
    }

    void Spin_System_Chain::Lock() const
    {
        try
        {
            this->mutex.lock();
            for (auto& image : this->images)
                image->Lock();
        }
        catch( ... )
        {
            spirit_handle_exception_core("Unlocking the Spin_System failed!");
        }
    }

    void Spin_System_Chain::Unlock() const
    {
        try
        {
            for (auto& image : this->images)
                image->Unlock();
            this->mutex.unlock();
        }
        catch( ... )
        {
            spirit_handle_exception_core("Unlocking the Spin_System failed!");
        }
    }
}