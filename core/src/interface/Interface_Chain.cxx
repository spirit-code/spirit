#include "Interface_Chain.h"
#include "Interface_State.h"
#include "Manifoldmath.h"

extern "C" int Chain_Get_Index(State * state)
{
    return state->idx_active_chain;
}

extern "C" int Chain_Get_NOI(State * state)
{
    return state->active_chain->noi;
}

extern "C" void Chain_next_Image(State * state, int idx_chain_i)
{
    int idx_image = -1, idx_chain = idx_chain_i;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    // Fetch correct indices and pointers
    from_indices(state, idx_image, idx_chain, image, chain);
    
    // Apply
    ++state->idx_active_image;
    state->active_image = chain->images[state->idx_active_image];
    chain->idx_active_image = state->idx_active_image;
}

extern "C" void Chain_prev_Image(State * state, int idx_chain_i)
{
    int idx_image = -1, idx_chain = idx_chain_i;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    // Fetch correct indices and pointers
    from_indices(state, idx_image, idx_chain, image, chain);

    // Apply
    --state->idx_active_image;
    state->active_image = chain->images[state->idx_active_image];
    chain->idx_active_image = state->idx_active_image;
}

extern "C" void Chain_Image_to_Clipboard(State * state, int idx_image_i, int idx_chain_i)
{
    int idx_image = idx_image_i, idx_chain = idx_chain_i;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    // Fetch correct indices and pointers
    from_indices(state, idx_image, idx_chain, image, chain);
    
    // Copy the image to clipboard
    state->clipboard_image = std::shared_ptr<Data::Spin_System>(new Data::Spin_System(*image));
    
    Log(Utility::Log_Level::INFO, Utility::Log_Sender::API, "Copied image " + std::to_string(idx_image) + " (chain " + std::to_string(idx_chain) + ") to clipboard");
}

extern "C" void Chain_Insert_Image_Before(State * state, int idx_image_i, int idx_chain_i)
{
    int idx_image = idx_image_i, idx_chain = idx_chain_i;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    // Fetch correct indices and pointers
    from_indices(state, idx_image, idx_chain, image, chain);

    if (state->clipboard_image.get())
    {
        // Copy the clipboard image
        auto copy = std::shared_ptr<Data::Spin_System>(new Data::Spin_System(*state->clipboard_image));
        
        // Add to chain
        chain->noi++;
        chain->images.insert(chain->images.begin() + idx_image, copy);
        chain->climbing_image.insert(chain->climbing_image.begin() + idx_image, false);
        chain->falling_image.insert(chain->falling_image.begin() + idx_image, false);

        // Update state
        state->noi++;
        state->active_image = state->active_chain->images[state->idx_active_image];
        // state->active_image = copy;
        // state->active_chain->idx_active_image = state->idx_active_image;

        // Update array lengths
        Chain_Setup_Data(state, idx_chain);

        Log(Utility::Log_Level::INFO, Utility::Log_Sender::API, "Inserted image before " + std::to_string(idx_image) + " (chain " + std::to_string(idx_chain) + ") from clipboard");
    }
    else
    {
        Log(Utility::Log_Level::L_ERROR, Utility::Log_Sender::API, "Tried to insert image before " + std::to_string(idx_image) + " (chain " + std::to_string(idx_chain) + ") but clipboard was empty");
    }
}

extern "C" void Chain_Insert_Image_After(State * state, int idx_image_i, int idx_chain_i)
{
    int idx_image = idx_image_i, idx_chain = idx_chain_i;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    // Fetch correct indices and pointers
    from_indices(state, idx_image, idx_chain, image, chain);
    
    if (state->clipboard_image.get())
    {
        // Copy the clipboard image
        auto copy = std::shared_ptr<Data::Spin_System>(new Data::Spin_System(*state->clipboard_image));
        
        // Add to chain
        chain->noi++;
        if (idx_image < state->noi - 1)
        {
            chain->images.insert(chain->images.begin() + idx_image + 1, copy);
            chain->climbing_image.insert(chain->climbing_image.begin() + idx_image + 1, false);
            chain->falling_image.insert(chain->falling_image.begin() + idx_image + 1, false);
        }
        else
        {
            chain->images.push_back(copy);
            chain->climbing_image.push_back(false);
            chain->falling_image.push_back(false);
        }
        // Update state
        state->noi++;
        state->active_image = state->active_chain->images[state->idx_active_image];
        // state->active_image = copy;
        // state->active_chain->idx_active_image = state->idx_active_image;

        // Update array lengths
        Chain_Setup_Data(state, idx_chain);

        Log(Utility::Log_Level::INFO, Utility::Log_Sender::API, "Inserted image after " + std::to_string(idx_image) + " (chain " + std::to_string(idx_chain) + ") from clipboard");
    }
    else
    {
        Log(Utility::Log_Level::L_ERROR, Utility::Log_Sender::API, "Tried to insert image after " + std::to_string(idx_image) + " (chain " + std::to_string(idx_chain) + ") but clipboard was empty");
    }
}

extern "C" void Chain_Replace_Image(State * state, int idx_image_i, int idx_chain_i)
{
    int idx_image = idx_image_i, idx_chain = idx_chain_i;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    // Fetch correct indices and pointers
    from_indices(state, idx_image, idx_chain, image, chain);
    
    if (state->clipboard_image.get())
    {
        // Copy the clipboard image
        auto copy = std::shared_ptr<Data::Spin_System>(new Data::Spin_System(*state->clipboard_image));
        
        // Replace in chain
        chain->images[idx_image] = copy;
        
        // Update state
        // state->noi++;
        state->active_image = state->active_chain->images[state->idx_active_image];
        // state->active_image = copy;
        // state->active_chain->idx_active_image = state->idx_active_image;
        Log(Utility::Log_Level::INFO, Utility::Log_Sender::API, "Replaced image " + std::to_string(idx_image) + " (chain " + std::to_string(idx_chain) + ") from clipboard");
    }
    else
    {
        Log(Utility::Log_Level::L_ERROR, Utility::Log_Sender::API, "Tried to replace image " + std::to_string(idx_image) + " (chain " + std::to_string(idx_chain) + ") but clipboard was empty");
    }
}

extern "C" void Chain_Delete_Image(State * state, int idx_image_i, int idx_chain_i)
{
    int idx_image = idx_image_i, idx_chain = idx_chain_i;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    // Fetch correct indices and pointers
    from_indices(state, idx_image, idx_chain, image, chain);

    // Apply
    state->noi--;
    
    chain->images.erase(chain->images.begin() + idx_image);
    chain->climbing_image.erase(chain->climbing_image.begin() + idx_image);
    chain->falling_image.erase(chain->falling_image.begin() + idx_image);

    state->active_image = state->active_chain->images[state->idx_active_image];
    state->active_chain->idx_active_image = state->idx_active_image;

    // Update array lengths
    Chain_Setup_Data(state, idx_chain);

    Log(Utility::Log_Level::INFO, Utility::Log_Sender::API, "Deleted image " + std::to_string(idx_image) + " (chain " + std::to_string(idx_chain) + ")");
}

extern "C" void Chain_Update_Data(State * state, int idx_chain_i)
{
    int idx_image = -1, idx_chain = idx_chain_i;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    // Fetch correct indices and pointers
    from_indices(state, idx_image, idx_chain, image, chain);

    // Apply
    for (int i = 0; i < state->noi; ++i)
    {
        //Engine::Energy::Update(*chain->images[i]);
        //chain->images[i]->E = chain->images[i]->hamiltonian_isotropichain->Energy(chain->images[i]->spins);
        chain->images[i]->UpdateEnergy();
        // if (i > 0) chain->Rx[i] = chain->Rx[i - 1] + Utility::Manifoldmath::Dist_Geodesic(chain->images[i - 1]->spins, state->active_chain->images[i]->spins);
    }
}

extern "C" void Chain_Setup_Data(State * state, int idx_chain_i)
{
    int idx_image = -1, idx_chain = idx_chain_i;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    // Fetch correct indices and pointers
    from_indices(state, idx_image, idx_chain, image, chain);

    // Apply
    chain->Rx = std::vector<double>(state->noi, 0);
    chain->Rx_interpolated = std::vector<double>((state->noi - 1)*chain->gneb_parameters->n_E_interpolations, 0);
    chain->E_interpolated = std::vector<double>((state->noi - 1)*chain->gneb_parameters->n_E_interpolations, 0);
    chain->E_array_interpolated = std::vector<std::vector<double>>(7, std::vector<double>((state->noi - 1)*chain->gneb_parameters->n_E_interpolations, 0));

    // Initial data update
    Chain_Update_Data(state, idx_chain);
}