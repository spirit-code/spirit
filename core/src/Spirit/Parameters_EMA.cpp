#include <Spirit/Parameters_EMA.h>

#include <data/Spin_System.hpp>
#include <data/Spin_System_Chain.hpp>
#include <data/State.hpp>
#include <utility/Exception.hpp>

#include <memory>

// Clears all the previously calculated modes from memory
void Parameters_EMA_Clear_Modes( State * state, int idx_image, int idx_chain ) noexcept
{

    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    Log( Utility::Log_Level::Info, Utility::Log_Sender::API, "Clearing modes", idx_image, idx_chain );

    image->lock();
    for( auto & el : image->modes )
    {
        el.reset();
    }
    image->unlock();
}

/*------------------------------------------------------------------------------------------------------ */
/*---------------------------------- Set EMA  ---------------------------------------------------------- */
/*------------------------------------------------------------------------------------------------------ */

// Set EMA Calculation Parameters
void Parameters_EMA_Set_N_Modes( State * state, int n_modes, int idx_image, int idx_chain ) noexcept
try
{

    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    if( n_modes < 1 || n_modes > 2 * image->nos )
    {
        Log( Utility::Log_Level::Debug, Utility::Log_Sender::API,
             fmt::format( "Illegal value of number of modes (max value is {})", 2 * image->nos ), idx_image,
             idx_chain );
    }
    else
    {
        image->lock();
        image->ema_parameters->n_modes = n_modes;
        image->modes.resize( n_modes );
        image->eigenvalues.resize( n_modes );
        image->ema_parameters->n_mode_follow = std::min( image->ema_parameters->n_mode_follow, n_modes );
        image->unlock();
    }
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Parameters_EMA_Set_N_Mode_Follow( State * state, int n_mode_follow, int idx_image, int idx_chain ) noexcept
try
{

    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    if( n_mode_follow < 0 || n_mode_follow > image->ema_parameters->n_modes - 1 || n_mode_follow >= image->modes.size()
        || !image->modes[n_mode_follow].has_value() )
    {
        Log( Utility::Log_Level::Debug, Utility::Log_Sender::API, fmt::format( "Illegal value of mode to follow" ),
             idx_image, idx_chain );
    }
    else
    {
        image->lock();
        image->ema_parameters->n_mode_follow = n_mode_follow;
        image->unlock();
    }
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Parameters_EMA_Set_Frequency( State * state, scalar frequency, int idx_image, int idx_chain ) noexcept
try
{

    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    image->lock();
    image->ema_parameters->frequency = frequency;
    image->unlock();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Parameters_EMA_Set_Amplitude( State * state, scalar amplitude, int idx_image, int idx_chain ) noexcept
try
{

    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    image->lock();
    image->ema_parameters->amplitude = amplitude;
    image->unlock();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Parameters_EMA_Set_Snapshot( State * state, bool snapshot, int idx_image, int idx_chain ) noexcept
try
{

    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    image->lock();
    image->ema_parameters->snapshot = snapshot;
    image->unlock();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Parameters_EMA_Set_Sparse( State * state, bool sparse, int idx_image, int idx_chain ) noexcept
try
{

    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    Log( Utility::Log_Level::Info, Utility::Log_Sender::API, fmt::format( "Setting parameter 'sparse' to {}", sparse ),
         idx_image, idx_chain );
    image->lock();
    image->ema_parameters->sparse = sparse;
    image->unlock();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

/*------------------------------------------------------------------------------------------------------ */
/*---------------------------------- Get EMA ----------------------------------------------------------- */
/*------------------------------------------------------------------------------------------------------ */

// Get EMA Calculation Parameters
int Parameters_EMA_Get_N_Modes( State * state, int idx_image, int idx_chain ) noexcept
try
{

    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    return image->ema_parameters->n_modes;
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return 0;
}

int Parameters_EMA_Get_N_Mode_Follow( State * state, int idx_image, int idx_chain ) noexcept
try
{

    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    return image->ema_parameters->n_mode_follow;
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return 0;
}

scalar Parameters_EMA_Get_Frequency( State * state, int idx_image, int idx_chain ) noexcept
try
{

    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    return image->ema_parameters->frequency;
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return 0;
}

scalar Parameters_EMA_Get_Amplitude( State * state, int idx_image, int idx_chain ) noexcept
try
{

    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    return image->ema_parameters->amplitude;
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return 0;
}

bool Parameters_EMA_Get_Snapshot( State * state, int idx_image, int idx_chain ) noexcept
try
{

    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    return image->ema_parameters->snapshot;
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return false;
}

bool Parameters_EMA_Get_Sparse( State * state, int idx_image, int idx_chain ) noexcept
try
{

    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    return image->ema_parameters->sparse;
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return false;
}
