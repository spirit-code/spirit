#include <data/Spin_System_Chain.hpp>
#include <utility/Exception.hpp>

namespace Data
{

Spin_System_Chain::Spin_System_Chain(
    const std::vector<std::shared_ptr<Spin_System>> & images,
    std::shared_ptr<Data::Parameters_Method_GNEB> gneb_parameters, bool iteration_allowed )
        : gneb_parameters( gneb_parameters ), iteration_allowed( iteration_allowed ), singleshot_allowed( false )
{
    this->noi    = images.size();
    this->images = images;

    this->idx_active_image = 0;

    this->image_type = std::vector<GNEB_Image_Type>( this->noi, GNEB_Image_Type::Normal );

    this->Rx                      = std::vector<scalar>( this->noi, 0 );
    std::size_t size_interpolated = this->noi + ( this->noi - 1 ) * gneb_parameters->n_E_interpolations;
    this->Rx_interpolated         = std::vector<scalar>( size_interpolated, 0 );
    this->E_interpolated          = std::vector<scalar>( size_interpolated, 0 );
    this->E_array_interpolated    = std::vector<std::vector<scalar>>( 7, std::vector<scalar>( size_interpolated, 0 ) );
}

void Spin_System_Chain::Lock() noexcept
try
{
    this->ordered_lock.lock();
    for( auto & image : this->images )
        image->Lock();
}
catch( ... )
{
    spirit_handle_exception_core( "Unlocking the Spin_System_Chain failed!" );
}

void Spin_System_Chain::Unlock() noexcept
try
{
    for( auto & image : this->images )
        image->Unlock();
    this->ordered_lock.unlock();
}
catch( ... )
{
    spirit_handle_exception_core( "Unlocking the Spin_System_Chain failed!" );
}

} // namespace Data