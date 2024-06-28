#include <Spirit/State.h>
#include <Spirit/System.h>

#include <data/State.hpp>
#include <engine/StateType.hpp>
#include <engine/spin/Eigenmodes.hpp>
#include <utility/Exception.hpp>
#include <utility/Logging.hpp>

using Engine::Field;
using Engine::get;

int System_Get_Index( State * state ) noexcept
try
{
    return state->idx_active_image;
}
catch( ... )
{
    spirit_handle_exception_api( -1, -1 );
    return -1;
}

int System_Get_NOS( State * state, int idx_image, int idx_chain ) noexcept
try
{

    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    return image->nos;
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return 0;
}

scalar * System_Get_Spin_Directions( State * state, int idx_image, int idx_chain ) noexcept
try
{

    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    return (scalar *)get<Field::Spin>( *image->state )[0].data();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return nullptr;
}

scalar * System_Get_Effective_Field( State * state, int idx_image, int idx_chain ) noexcept
try
{

    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );
    return image->M.effective_field[0].data();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return nullptr;
}

scalar * System_Get_Eigenmode( State * state, int idx_mode, int idx_image, int idx_chain ) noexcept
try
{

    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    // Check mode index
    if( idx_mode >= image->modes.size() )
    {
        Log( Utility::Log_Level::Error, Utility::Log_Sender::API,
             fmt::format( "Invalid mode index {}, image has only {} modes stored.", idx_mode, image->modes.size() ) );
        return nullptr;
    }

    // Check if mode has been calculated
    if( !image->modes[idx_mode].has_value() )
    {
        Log( Utility::Log_Level::Error, Utility::Log_Sender::API,
             fmt::format( "Mode {} has not yet been calculated.", idx_mode ) );
        return nullptr;
    }

    return ( *image->modes[idx_mode] )[0].data();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return nullptr;
}

scalar System_Get_Rx( State * state, int idx_image, int idx_chain ) noexcept
try
{

    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    return chain->Rx[idx_image];
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return 0;
}

scalar System_Get_Energy( State * state, int idx_image, int idx_chain ) noexcept
try
{

    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    return image->E.total;
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return 0;
}

int System_Get_Energy_Array_Names( State * state, char * names, int idx_image, int idx_chain ) noexcept
try
{

    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    int n_char_array = -1; // Start of with offset -1, because the last contributions gets no "|" delimiter
    for( unsigned int i = 0; i < image->E.per_interaction.size(); ++i )
    {
        n_char_array += image->E.per_interaction[i].first.size()
                        + 1; // Add +1 because we separate the contribution names with the character "|"
    }

    // If 'names' is a nullptr, we return the required length of the names array
    if( names == nullptr )
    {
        return n_char_array;
    }
    else
    { // Else we try to fill the provided char array
        int idx = 0;
        for( unsigned int i = 0; i < image->E.per_interaction.size(); ++i )
        {
            for( const char & cur_char : ( image->E.per_interaction[i] ).first )
            {
                names[idx++] = cur_char;
            }
            if( i != image->E.per_interaction.size() - 1 )
                names[idx++] = '|';
        }
        return -1;
    }
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return -1;
}

int System_Get_Energy_Array(
    State * state, scalar * energies, bool divide_by_nspins, int idx_image, int idx_chain ) noexcept
try
{

    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    scalar nd = divide_by_nspins ? 1 / scalar( image->nos ) : 1;

    if( energies == nullptr )
    {
        return image->E.per_interaction.size();
    }
    else
    {
        for( unsigned int i = 0; i < image->E.per_interaction.size(); ++i )
        {
            energies[i] = nd * image->E.per_interaction[i].second;
        }
        return -1;
    }
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return -1;
}

void System_Get_Eigenvalues( State * state, scalar * eigenvalues, int idx_image, int idx_chain ) noexcept
try
{

    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    for( unsigned int i = 0; i < image->eigenvalues.size(); ++i )
    {
        eigenvalues[i] = image->eigenvalues[i];
    }
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void System_Print_Energy_Array( State * state, int idx_image, int idx_chain ) noexcept
try
{

    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    scalar nd = 1 / (scalar)image->nos;

    std::cerr << "E_tot = " << image->E.total * nd << "  ||  ";

    for( unsigned int i = 0; i < image->E.per_interaction.size(); ++i )
    {
        std::cerr << image->E.per_interaction[i].first << " = " << image->E.per_interaction[i].second * nd;
        if( i < image->E.per_interaction.size() - 1 )
            std::cerr << "  |  ";
    }
    std::cerr << std::endl;
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void System_Update_Data( State * state, int idx_image, int idx_chain ) noexcept
try
{

    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    image->lock();
    try
    {
        image->UpdateEnergy();
    }
    catch( ... )
    {
        spirit_handle_exception_api( idx_image, idx_chain );
    }
    image->unlock();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

// Calculate the eigenmodes of the System
void System_Update_Eigenmodes( State * state, int idx_image, int idx_chain ) noexcept
try
{

    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    image->lock();
    Engine::Spin::Eigenmodes::Calculate_Eigenmodes( *image, idx_image, idx_chain );
    image->unlock();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}
