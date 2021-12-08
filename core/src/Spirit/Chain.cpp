#include <Spirit/Chain.h>
#include <Spirit/Simulation.h>
#include <Spirit/State.h>

#include <data/State.hpp>
#include <engine/Manifoldmath.hpp>
#include <engine/Vectormath.hpp>
#include <utility/Exception.hpp>
#include <utility/Logging.hpp>

#include <fmt/format.h>

int Chain_Get_NOI( State * state, int idx_chain ) noexcept
try
{
    return state->chain->noi;
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
    return 0;
}

bool Chain_next_Image( State * state, int idx_chain ) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    // Apply
    ++chain->idx_active_image;
    State_Update( state );

    Log( Utility::Log_Level::Debug, Utility::Log_Sender::API,
         fmt::format( "Switched to next image {} of {}", chain->idx_active_image + 1, chain->noi ),
         chain->idx_active_image, idx_chain );

    return true;
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
    return false;
}

bool Chain_prev_Image( State * state, int idx_chain ) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    // Apply
    if( idx_image > 0 )
    {
        --chain->idx_active_image;
        State_Update( state );
        Log( Utility::Log_Level::Debug, Utility::Log_Sender::API,
             fmt::format( "Switched to previous image {} of {}", chain->idx_active_image + 1, chain->noi ),
             chain->idx_active_image, idx_chain );
        return true;
    }
    else
    {
        Log( Utility::Log_Level::Error, Utility::Log_Sender::API, "Tried to switch to previous image.",
             chain->idx_active_image, idx_chain );
        return false;
    }
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
    return false;
}

bool Chain_Jump_To_Image( State * state, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    chain->idx_active_image = idx_image;
    State_Update( state );

    Log( Utility::Log_Level::Debug, Utility::Log_Sender::API,
         fmt::format( "Jumped to image {} of {}", chain->idx_active_image + 1, chain->noi ), idx_image, idx_chain );

    return true;
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return false;
}

void Chain_Set_Length( State * state, int n_images, int idx_chain ) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    if( n_images < 1 )
    {
        Log( Utility::Log_Level::Warning, Utility::Log_Sender::API, "Tried to reduce length of chain below 1...", -1,
             idx_chain );
        return;
    }

    if( n_images == chain->noi )
        return;

    if( Simulation_Running_On_Chain( state, idx_chain ) )
    {
        chain->iteration_allowed = false;
        Simulation_Stop( state, idx_image, idx_chain );
    }

    // Increase the chain length
    if( n_images > chain->noi )
    {
        if( !state->clipboard_image )
        {
            if( chain->noi > 1 )
            {
                // This message is only relevant if there is more than 1 image
                Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
                     fmt::format( "Clipboard was empty, so the {}. image will be used.", idx_image ), -1, idx_chain );
            }
            Chain_Image_to_Clipboard( state );
        }

        for( idx_image = chain->noi - 1; idx_image < n_images - 1; ++idx_image )
        {
            // Copy the clipboard image
            state->clipboard_image->Lock();
            auto copy = std::shared_ptr<Data::Spin_System>( new Data::Spin_System( *state->clipboard_image ) );
            state->clipboard_image->Unlock();

            chain->Lock();
            copy->Lock();

            // Add to chain
            chain->noi++;
            chain->images.push_back( copy );
            chain->image_type.push_back( Data::GNEB_Image_Type::Normal );

            // Add to state
            state->method_image.push_back( std::shared_ptr<Engine::Method>() );

            chain->Unlock();
        }

        // Update state
        State_Update( state );

        // Update array lengths
        Chain_Setup_Data( state, idx_chain );

        Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
             fmt::format( "Increased length of chain to {}", chain->noi ), -1, idx_chain );
    }
    // Reduce the chain length
    else if( n_images < chain->noi )
    {
        for( idx_image = chain->noi - 1; idx_image > n_images - 1; --idx_image )
        {
            // Stop any simulations running on an image we want to remove
            Simulation_Stop( state, idx_image, idx_chain );

            chain->Lock();
            try
            {
                // Remove from chain
                chain->noi--;
                if( chain->idx_active_image == chain->noi )
                    Chain_prev_Image( state, idx_chain );

                state->noi = state->chain->noi;

                chain->images.back()->Unlock();
                chain->images.pop_back();
                chain->image_type.pop_back();

                // Add to state
                state->method_image.pop_back();
            }
            catch( ... )
            {
                spirit_handle_exception_api( idx_image, idx_chain );
            }
            chain->Unlock();
        }
        // Update state
        State_Update( state );

        // Update array lengths
        Chain_Setup_Data( state, idx_chain );

        Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
             fmt::format( "Reduced length of chain to {}", chain->noi ), -1, idx_chain );
    }
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
}

void Chain_Image_to_Clipboard( State * state, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    // Copy the image to clipboard
    image->Lock();
    try
    {
        state->clipboard_image = std::shared_ptr<Data::Spin_System>( new Data::Spin_System( *image ) );
    }
    catch( ... )
    {
        spirit_handle_exception_api( idx_image, idx_chain );
    }
    image->Unlock();

    Log( Utility::Log_Level::Info, Utility::Log_Sender::API, "Copied image to clipboard.", idx_image, idx_chain );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Chain_Replace_Image( State * state, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    if( state->clipboard_image )
    {
        // Copy the clipboard image
        state->clipboard_image->Lock();
        auto copy = std::shared_ptr<Data::Spin_System>( new Data::Spin_System( *state->clipboard_image ) );
        state->clipboard_image->Unlock();

        chain->Lock();
        copy->Lock();

        // Replace in chain
        chain->images[idx_image]->Unlock();
        chain->images[idx_image] = copy;

        // Update state
        state->active_image = state->chain->images[state->idx_active_image];

        chain->Unlock();

        Log( Utility::Log_Level::Info, Utility::Log_Sender::API, "Replaced image.", idx_image, idx_chain );
    }
    else
    {
        Log( Utility::Log_Level::Info, Utility::Log_Sender::API, "Tried to replace image, but clipboard was empty.",
             idx_image, idx_chain );
    }
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Chain_Insert_Image_Before( State * state, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    if( state->clipboard_image )
    {
        if( Simulation_Running_On_Chain( state, idx_chain ) )
        {
            chain->iteration_allowed = false;
            Simulation_Stop( state, idx_image, idx_chain );
        }

        // Copy the clipboard image
        state->clipboard_image->Lock();
        auto copy = std::shared_ptr<Data::Spin_System>( new Data::Spin_System( *state->clipboard_image ) );
        state->clipboard_image->Unlock();

        chain->Lock();
        copy->Lock();

        // Add to chain
        chain->noi++;
        chain->images.insert( chain->images.begin() + idx_image, copy );
        chain->image_type.insert( chain->image_type.begin() + idx_image, Data::GNEB_Image_Type::Normal );

        // Add to state
        state->method_image.insert( state->method_image.begin() + idx_image, std::shared_ptr<Engine::Method>() );

        // Increment active image so that we don't switch between images
        ++chain->idx_active_image;

        chain->Unlock();

        // Update state
        State_Update( state );

        // Update array lengths
        Chain_Setup_Data( state, idx_chain );

        Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
             fmt::format( "Inserted image before. NOI is now {}", chain->noi ), idx_image, idx_chain );
    }
    else
    {
        Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
             "Tried to insert image before, but clipboard was empty.", idx_image, idx_chain );
    }
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Chain_Insert_Image_After( State * state, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    if( state->clipboard_image )
    {
        if( Simulation_Running_On_Chain( state, idx_chain ) )
        {
            chain->iteration_allowed = false;
            Simulation_Stop( state, idx_image, idx_chain );
        }

        // Copy the clipboard image
        state->clipboard_image->Lock();
        auto copy = std::shared_ptr<Data::Spin_System>( new Data::Spin_System( *state->clipboard_image ) );
        state->clipboard_image->Unlock();

        chain->Lock();
        copy->Lock();

        // Add to chain
        chain->noi++;
        chain->images.insert( chain->images.begin() + idx_image + 1, copy );
        chain->image_type.insert( chain->image_type.begin() + idx_image + 1, Data::GNEB_Image_Type::Normal );

        // Add to state
        state->method_image.insert( state->method_image.begin() + idx_image + 1, std::shared_ptr<Engine::Method>() );

        chain->Unlock();

        // Update state
        State_Update( state );

        // Update array lengths
        Chain_Setup_Data( state, idx_chain );

        Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
             fmt::format( "Inserted image after. NOI is now {}", chain->noi ), idx_image, idx_chain );
    }
    else
    {
        Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
             "Tried to insert image after, but clipboard was empty.", idx_image, idx_chain );
    }
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Chain_Push_Back( State * state, int idx_chain ) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    if( state->clipboard_image )
    {
        if( Simulation_Running_On_Chain( state, idx_chain ) )
        {
            chain->iteration_allowed = false;
            Simulation_Stop( state, idx_image, idx_chain );
        }

        // Copy the clipboard image
        state->clipboard_image->Lock();
        auto copy = std::shared_ptr<Data::Spin_System>( new Data::Spin_System( *state->clipboard_image ) );
        state->clipboard_image->Unlock();

        chain->Lock();
        copy->Lock();

        // Add to chain
        chain->noi++;
        chain->images.push_back( copy );
        chain->image_type.push_back( Data::GNEB_Image_Type::Normal );

        // Add to state
        state->method_image.push_back( std::shared_ptr<Engine::Method>() );

        chain->Unlock();

        // Update state
        State_Update( state );

        // Update array lengths
        Chain_Setup_Data( state, idx_chain );

        Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
             fmt::format( "Pushed back image from clipboard to chain. NOI is now {}", chain->noi ), -1, idx_chain );
    }
    else
    {
        Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
             "Tried to push back image to chain, but clipboard was empty.", -1, idx_chain );
    }
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
}

bool Chain_Delete_Image( State * state, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    // Apply
    if( chain->noi > 1 )
    {
        // Stop any simulations running on an image we want to remove
        Simulation_Stop( state, idx_image, idx_chain );

        chain->Lock();
        try
        {
            // Remove from chain
            chain->noi--;
            if( chain->idx_active_image == chain->noi )
                Chain_prev_Image( state, idx_chain );

            state->noi = state->chain->noi;

            chain->images[idx_image]->Unlock();
            chain->images.erase( chain->images.begin() + idx_image );
            chain->image_type.erase( chain->image_type.begin() + idx_image );

            // Remove from state
            state->method_image.erase( state->method_image.begin() + idx_image );
        }
        catch( ... )
        {
            spirit_handle_exception_api( idx_image, idx_chain );
        }
        chain->Unlock();

        // Update State
        State_Update( state );

        // Update array lengths
        Chain_Setup_Data( state, idx_chain );

        Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
             fmt::format( "Deleted image {} of {}", idx_image + 1, chain->noi + 1 ), -1, idx_chain );

        return true;
    }
    else
    {
        Log( Utility::Log_Level::Warning, Utility::Log_Sender::API, "Tried to delete last image.", 0, idx_chain );
        return false;
    }
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return false;
}

bool Chain_Pop_Back( State * state, int idx_chain ) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    if( chain->noi > 1 )
    {
        // Stop any simulations running on an image we want to remove
        Simulation_Stop( state, idx_image, idx_chain );

        chain->Lock();
        try
        {
            // Remove from chain
            chain->noi--;
            if( chain->idx_active_image == chain->noi )
                Chain_prev_Image( state, idx_chain );

            state->noi = state->chain->noi;

            chain->images.back()->Unlock();
            chain->images.pop_back();
            chain->image_type.pop_back();

            // Remove from state
            state->method_image.pop_back();
        }
        catch( ... )
        {
            spirit_handle_exception_api( idx_image, idx_chain );
        }
        chain->Unlock();

        // Update state
        State_Update( state );

        // Update array lengths
        Chain_Setup_Data( state, idx_chain );

        Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
             fmt::format( "Popped back image of chain. NOI is now {}", chain->noi ), -1, idx_chain );

        return true;
    }
    else
    {
        Log( Utility::Log_Level::Warning, Utility::Log_Sender::API, "Tried to delete last image.", 0, idx_chain );
        return false;
    }
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
    return false;
}

void Chain_Get_Rx( State * state, float * Rx, int idx_chain ) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    for( unsigned int i = 0; i < chain->Rx.size(); ++i )
    {
        Rx[i] = (float)chain->Rx[i];
    }
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
}

void Chain_Get_Rx_Interpolated( State * state, float * Rx_interpolated, int idx_chain ) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    for( unsigned int i = 0; i < chain->Rx_interpolated.size(); ++i )
    {
        Rx_interpolated[i] = (float)chain->Rx_interpolated[i];
    }
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
}

void Chain_Get_Energy( State * state, float * Energy, int idx_chain ) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    for( int i = 0; i < chain->noi; ++i )
    {
        Energy[i] = (float)chain->images[i]->E;
    }
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
}

void Chain_Get_Energy_Interpolated( State * state, float * E_interpolated, int idx_chain ) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    for( unsigned int i = 0; i < chain->E_interpolated.size(); ++i )
    {
        E_interpolated[i] = (float)chain->E_interpolated[i];
    }
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
}

std::vector<std::vector<float>> Chain_Get_Energy_Array_Interpolated( State * state, int idx_chain ) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    std::vector<std::vector<float>> E_arr_interpolated( chain->E_array_interpolated.size() );
    for( unsigned int i = 0; i < chain->E_array_interpolated.size(); i++ )
        E_arr_interpolated[i] = std::vector<float>( chain->E_array_interpolated[i].size(), 0 );

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    for( unsigned int i = 0; i < chain->E_array_interpolated.size(); i++ )
        for( unsigned int j = 0; j < chain->E_array_interpolated[i].size(); j++ )
            E_arr_interpolated[i][j] = (float)chain->E_array_interpolated[i][j];

    return E_arr_interpolated;
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );

    // XXX: what should we return in that situation
    std::vector<std::vector<float>> E_at_failure( 1 );
    E_at_failure[0] = std::vector<float>( 1, 0 );
    return E_at_failure;
}

void Chain_Update_Data( State * state, int idx_chain ) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    // Apply
    for( int i = 0; i < chain->noi; ++i )
    {
        // Engine::Energy::Update(*chain->images[i]);
        // chain->images[i]->E = chain->images[i]->hamiltonian_isotropichain->Energy(chain->images[i]->spins);

        chain->images[i]->Lock();
        try
        {
            chain->images[i]->UpdateEnergy();
            if( i > 0 )
                chain->Rx[i]
                    = chain->Rx[i - 1]
                      + Engine::Manifoldmath::dist_geodesic( *chain->images[i - 1]->spins, *chain->images[i]->spins );
        }
        catch( ... )
        {
            spirit_handle_exception_api( idx_image, idx_chain );
        }
        chain->images[i]->Unlock();
    }
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
}

void Chain_Setup_Data( State * state, int idx_chain ) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    chain->Lock();

    try
    {
        // Apply
        chain->Rx = std::vector<scalar>( state->noi, 0 );
        chain->Rx_interpolated
            = std::vector<scalar>( state->noi + ( state->noi - 1 ) * chain->gneb_parameters->n_E_interpolations, 0 );
        chain->E_interpolated
            = std::vector<scalar>( state->noi + ( state->noi - 1 ) * chain->gneb_parameters->n_E_interpolations, 0 );
        chain->E_array_interpolated = std::vector<std::vector<scalar>>(
            7, std::vector<scalar>( state->noi + ( state->noi - 1 ) * chain->gneb_parameters->n_E_interpolations, 0 ) );
    }
    catch( ... )
    {
        spirit_handle_exception_api( idx_image, idx_chain );
    }

    chain->Unlock();

    // Initial data update
    Chain_Update_Data( state, idx_chain );
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
}