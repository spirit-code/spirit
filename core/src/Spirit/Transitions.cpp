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
{
    int idx_image = -1;
    
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        // Use this when State implements chain collection: else c = state->collection[idx_chain];
        chain->Lock();
        try
        {
            Utility::Configuration_Chain::Homogeneous_Rotation(chain, idx_1, idx_2);
            for (int img = 0; img < chain->noi; ++img)
            {
                chain->images[img]->geometry->Apply_Pinning(*chain->images[img]->spins);
            }
        }
        catch( ... )
        {
            spirit_handle_exception_api(idx_image, idx_chain);
        }
        chain->Unlock();

        Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
             fmt::format("Set homogeneous transition between images {} and {}", idx_1+1, idx_2+1), -1, idx_chain );
    }
    catch( ... )
    {
    spirit_handle_exception_api(idx_image, idx_chain);
    }
}

void Transition_Add_Noise_Temperature( State *state, float temperature, int idx_1, int idx_2, int idx_chain ) noexcept
{
    int idx_image = -1;
    
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        // Use this when State implements chain collection: else c = state->collection[idx_chain];
        chain->Lock();
        try
        {
            Utility::Configuration_Chain::Add_Noise_Temperature(chain, idx_1, idx_2, temperature);
            for (int img = 0; img < chain->noi; ++img)
            {
                chain->images[img]->geometry->Apply_Pinning(*chain->images[img]->spins);
            }
        }
        catch( ... )
        {
            spirit_handle_exception_api(idx_image, idx_chain);
        }
        chain->Unlock();

        Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
            fmt::format("Added noise with temperature T={} to images {} - {}", temperature, idx_1+1, idx_2+1 ), -1, idx_chain );
    }
    catch( ... )
    {
        spirit_handle_exception_api(-1, idx_chain);
    }
}