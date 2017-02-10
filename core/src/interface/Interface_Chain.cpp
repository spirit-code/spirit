#include <Spirit/Chain.h>
#include <Spirit/State.h>
#include <data/State.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Manifoldmath.hpp>
#include <utility/Logging.hpp>

int Chain_Get_Index(State * state)
{
    return state->idx_active_chain;
}

int Chain_Get_NOI(State * state, int idx_chain)
{
    if (idx_chain>=0) return state->collection->chains[idx_chain]->noi;
    return state->active_chain->noi;
}

bool Chain_next_Image(State * state, int idx_chain_i)
{
    int idx_image = -1, idx_chain = idx_chain_i;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    // Fetch correct indices and pointers
    from_indices(state, idx_image, idx_chain, image, chain);
    
    // Apply
    if ( idx_image < chain->noi-1 )
    {
        ++chain->idx_active_image;
        State_Update(state);
        Log(Utility::Log_Level::Debug, Utility::Log_Sender::API,
            "Switched to next image " + std::to_string(chain->idx_active_image+1) + " of " + std::to_string(chain->noi), chain->idx_active_image, idx_chain);
        return true;
    }
    else
    {
        Log(Utility::Log_Level::Error, Utility::Log_Sender::API,
            "Tried to switch to next image.", chain->idx_active_image, idx_chain);
        return false;
    }
}

bool Chain_prev_Image(State * state, int idx_chain_i)
{
    int idx_image = -1, idx_chain = idx_chain_i;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    // Fetch correct indices and pointers
    from_indices(state, idx_image, idx_chain, image, chain);

    // Apply
    if ( idx_image > 0 )
    {
        --chain->idx_active_image;
        State_Update(state);
        Log(Utility::Log_Level::Debug, Utility::Log_Sender::API,
            "Switched to previous image " + std::to_string(chain->idx_active_image+1) + " of " + std::to_string(chain->noi), chain->idx_active_image, idx_chain);
        return true;
    }
    else
    {
        Log(Utility::Log_Level::Error, Utility::Log_Sender::API,
            "Tried to switch to previous image.", chain->idx_active_image, idx_chain);
        return false;
    }
}

bool Chain_Jump_To_Image(State * state, int idx_image_i, int idx_chain_i)
{
    int idx_image = idx_image_i, idx_chain = idx_chain_i;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    // Fetch correct indices and pointers
    from_indices(state, idx_image, idx_chain, image, chain);

    // Apply
    if ( idx_image >= 0 && idx_image < chain->noi )
    {
        chain->idx_active_image = idx_image;
        State_Update(state);
        Log(Utility::Log_Level::Debug, Utility::Log_Sender::API,
            "Jumped to image " + std::to_string(chain->idx_active_image+1) + " of " + std::to_string(chain->noi), idx_image, idx_chain);
        return true;
    }
    else
    {
        Log(Utility::Log_Level::Error, Utility::Log_Sender::API,
            "Tried to jump to image " + std::to_string(idx_image+1) + " of " + std::to_string(chain->noi));
        return false;
    }
}

void Chain_Image_to_Clipboard(State * state, int idx_image_i, int idx_chain_i)
{
    int idx_image = idx_image_i, idx_chain = idx_chain_i;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    // Fetch correct indices and pointers
    from_indices(state, idx_image, idx_chain, image, chain);
    
    // Copy the image to clipboard
    state->clipboard_image = std::shared_ptr<Data::Spin_System>(new Data::Spin_System(*image));
    
    Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
        "Copied image to clipboard.", idx_image, idx_chain);
}

void Chain_Replace_Image(State * state, int idx_image_i, int idx_chain_i)
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
        state->active_image = state->active_chain->images[state->idx_active_image];
        Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
            "Replaced image.", idx_image, idx_chain);
    }
    else
    {
        Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
            "Tried to replace image, but clipboard was empty.", idx_image, idx_chain);
    }
}

void Chain_Insert_Image_Before(State * state, int idx_image_i, int idx_chain_i)
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
		chain->image_type.insert(chain->image_type.begin() + idx_image, Data::GNEB_Image_Type::Normal);

		// Add to state
		state->simulation_information_llg[idx_chain].insert(state->simulation_information_llg[idx_chain].begin() + idx_image, std::shared_ptr<Simulation_Information>());

        // Increment active image so that we don't switch between images
        ++chain->idx_active_image;

        // Update state
        State_Update(state);

        // Update array lengths
        Chain_Setup_Data(state, idx_chain);

        Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
            "Inserted image before. NOI is now " + std::to_string(chain->noi), idx_image, idx_chain);
    }
    else
    {
        Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
            "Tried to insert image before, but clipboard was empty.", idx_image, idx_chain);
    }
}

void Chain_Insert_Image_After(State * state, int idx_image_i, int idx_chain_i)
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
        // if (idx_image < state->noi - 1)
        // {
            chain->images.insert(chain->images.begin() + idx_image + 1, copy);
			chain->image_type.insert(chain->image_type.begin() + idx_image + 1, Data::GNEB_Image_Type::Normal);
        // }
        // else
        // {
        //     chain->images.push_back(copy);
        //     chain->climbing_image.push_back(false);
        //     chain->falling_image.push_back(false);
        // }

		// Add to state
		state->simulation_information_llg[idx_chain].insert(state->simulation_information_llg[idx_chain].begin() + idx_image, std::shared_ptr<Simulation_Information>());

        // Update state
        State_Update(state);

        // Update array lengths
        Chain_Setup_Data(state, idx_chain);

        Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
            "Inserted image after. NOI is now " + std::to_string(chain->noi), idx_image, idx_chain);
    }
    else
    {
        Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
            "Tried to insert image after, but clipboard was empty.", idx_image, idx_chain);
    }
}


void Chain_Push_Back(State * state, int idx_chain_i)
{
    int idx_image = -1, idx_chain = idx_chain_i;
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
        chain->images.push_back(copy);
        chain->image_type.push_back(Data::GNEB_Image_Type::Normal);
            
		// Add to state
		state->simulation_information_llg[idx_chain].push_back(std::shared_ptr<Simulation_Information>());

        // Update state
        State_Update(state);

        // Update array lengths
        Chain_Setup_Data(state, idx_chain);

        Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
            "Pushed back image from clipboard to chain. NOI is now " + std::to_string(chain->noi), -1, idx_chain);
    }
    else
    {
        Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
            "Tried to push back image to chain, but clipboard was empty.", -1, idx_chain);
    }
}

bool Chain_Delete_Image(State * state, int idx_image_i, int idx_chain_i)
{
    int idx_image = idx_image_i, idx_chain = idx_chain_i;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    // Fetch correct indices and pointers
    from_indices(state, idx_image, idx_chain, image, chain);

    // Apply
    if (chain->noi > 1)
    {
        chain->noi--;
        state->noi = state->active_chain->noi;
        
        chain->images.erase(chain->images.begin() + idx_image);
        chain->image_type.erase(chain->image_type.begin() + idx_image);

		// Remove from state
		state->simulation_information_llg[idx_chain].erase(state->simulation_information_llg[idx_chain].begin() + idx_image);

		// Update State
        State_Update(state);

        // Update array lengths
        Chain_Setup_Data(state, idx_chain);

        Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
            "Deleted image " + std::to_string(idx_image+1) + " of " + std::to_string(chain->noi+1), -1, idx_chain);
        return true;
    }
    else
    {
        Log(Utility::Log_Level::Warning, Utility::Log_Sender::API,
            "Tried to delete last image.", 0, idx_chain);
        return false;
    }
}

bool Chain_Pop_Back(State * state, int idx_chain_i)
{
    int idx_image = -1, idx_chain = idx_chain_i;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    // Fetch correct indices and pointers
    from_indices(state, idx_image, idx_chain, image, chain);
    
    
    if (chain->noi > 1)
    {
        // Add to chain
        chain->noi--;
        state->noi = state->active_chain->noi;

        chain->images.pop_back();
        chain->image_type.pop_back();
            
        // Add to state
        state->simulation_information_llg[idx_chain].pop_back();

        // Update state
        State_Update(state);

        // Update array lengths
        Chain_Setup_Data(state, idx_chain);

        Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
            "Popped back image of chain. NOI is now " + std::to_string(chain->noi), -1, idx_chain);
		return true;
    }
    else
    {
        Log(Utility::Log_Level::Warning, Utility::Log_Sender::API,
            "Tried to delete last image.", 0, idx_chain);
        return false;
    }
}

void Chain_Get_Rx(State * state, float * Rx, int idx_chain)
{
	int idx_image = -1;
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

	for (unsigned int i = 0; i < chain->Rx.size(); ++i)
	{
		Rx[i] = (float)chain->Rx[i];
	}
}

void Chain_Get_Rx_Interpolated(State * state, float * Rx_interpolated, int idx_chain)
{
	int idx_image = -1;
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

	for (unsigned int i = 0; i < chain->Rx_interpolated.size(); ++i)
	{
		Rx_interpolated[i] = (float)chain->Rx_interpolated[i];
	}
}

void Chain_Get_Energy(State * state, float * Energy, int idx_chain)
{
	int idx_image = -1;
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

	for (int i = 0; i < chain->noi; ++i)
	{
		Energy[i] = (float)chain->images[i]->E;
	}
}

void Chain_Get_Energy_Interpolated(State * state, float * E_interpolated, int idx_chain)
{
	int idx_image = -1;
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

	for (unsigned int i = 0; i < chain->E_interpolated.size(); ++i)
	{
		E_interpolated[i] = (float)chain->E_interpolated[i];
	}
}

std::vector<std::vector<float>> Chain_Get_Energy_Array_Interpolated(State * state, int idx_chain_i)
{
	int idx_image = -1, idx_chain = idx_chain_i;
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	// Fetch correct indices and pointers
	from_indices(state, idx_image, idx_chain, image, chain);

	std::vector<std::vector<float>> E_array_interpolated(chain->E_array_interpolated.size());
	for (unsigned int i = 0; i < chain->E_array_interpolated.size(); ++i)
	{
		E_array_interpolated[i] = std::vector<float>(chain->E_array_interpolated[i].size());
		for (unsigned j = 0; j < chain->E_array_interpolated[i].size(); ++j)
		{
			E_array_interpolated[i][j] = (float)chain->E_array_interpolated[i][j];
		}
	}
	return E_array_interpolated;
}



void Chain_Update_Data(State * state, int idx_chain_i)
{
    int idx_image = -1, idx_chain = idx_chain_i;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    // Fetch correct indices and pointers
    from_indices(state, idx_image, idx_chain, image, chain);

    // Apply
    for (int i = 0; i < chain->noi; ++i)
    {
        //Engine::Energy::Update(*chain->images[i]);
        //chain->images[i]->E = chain->images[i]->hamiltonian_isotropichain->Energy(chain->images[i]->spins);
        chain->images[i]->UpdateEnergy();
        if (i > 0) chain->Rx[i] = chain->Rx[i-1] + Engine::Manifoldmath::dist_geodesic(*chain->images[i-1]->spins, *chain->images[i]->spins);
    }
}

void Chain_Setup_Data(State * state, int idx_chain_i)
{
    int idx_image = -1, idx_chain = idx_chain_i;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    // Fetch correct indices and pointers
    from_indices(state, idx_image, idx_chain, image, chain);

    // Apply
    chain->Rx = std::vector<scalar>(state->noi, 0);
    chain->Rx_interpolated = std::vector<scalar>(state->noi + (state->noi - 1)*chain->gneb_parameters->n_E_interpolations, 0);
    chain->E_interpolated = std::vector<scalar>(state->noi + (state->noi - 1)*chain->gneb_parameters->n_E_interpolations, 0);
    chain->E_array_interpolated = std::vector<std::vector<scalar>>(7, std::vector<scalar>(state->noi + (state->noi - 1)*chain->gneb_parameters->n_E_interpolations, 0));

    // Initial data update
    Chain_Update_Data(state, idx_chain);
}