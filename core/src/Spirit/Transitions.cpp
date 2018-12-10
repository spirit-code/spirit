#include <Spirit/Transitions.h>
#include <Spirit/State.h>
#include <data/State.hpp>
#include <data/Spin_System_Chain.hpp>
#include <utility/Configuration_Chain.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>

#include <fmt/format.h>

#include <memory>

void Transition_Homogeneous(State *state, int idx_1, int idx_2, int idx_chain) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    // Check indices
    if( idx_2 <= idx_1 )
    {
        Log( Utility::Log_Level::Error, Utility::Log_Sender::API, fmt::format(
            "Cannot set homogeneous transition between images {} and {}, because the second index needs to be larger than the first",
            idx_1+1, idx_2+1), -1, idx_chain );
        return;
    }

    chain->Lock();
    try
    {
        Utility::Configuration_Chain::Homogeneous_Rotation(chain, idx_1, idx_2);
        for (int img = 0; img < chain->noi; ++img)
            chain->images[img]->geometry->Apply_Pinning(*chain->images[img]->spins);

        Log( Utility::Log_Level::Info, Utility::Log_Sender::API, fmt::format(
            "Set homogeneous transition between images {} and {}",
            idx_1+1, idx_2+1), -1, idx_chain );
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
    }
    chain->Unlock();
}
catch( ... )
{
    spirit_handle_exception_api(-1, idx_chain);
}

void Transition_Add_Noise_Temperature( State *state, float temperature, int idx_1, int idx_2, int idx_chain ) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    // Check indices
    if( idx_2 <= idx_1 )
    {
        Log( Utility::Log_Level::Error, Utility::Log_Sender::API, fmt::format(
            "Cannot set homogeneous transition between images {} and {}, because the second index needs to be larger than the first",
            idx_1+1, idx_2+1), -1, idx_chain );
        return;
    }

    chain->Lock();
    try
    {
        Utility::Configuration_Chain::Add_Noise_Temperature(chain, idx_1, idx_2, temperature);
        for( int img = 0; img < chain->noi; ++img )
            chain->images[img]->geometry->Apply_Pinning(*chain->images[img]->spins);

        Log( Utility::Log_Level::Info, Utility::Log_Sender::API, fmt::format(
            "Added noise with temperature T={} to images {} - {}",
            temperature, idx_1+1, idx_2+1 ), -1, idx_chain );
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
    }
    chain->Unlock();
}
catch( ... )
{
    spirit_handle_exception_api(-1, idx_chain);
}