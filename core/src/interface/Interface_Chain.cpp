#include "Interface_Chain.h"
#include "Interface_State.h"

#include "State.hpp"
#include "engine/Vectormath.hpp"

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
        return true;
    }
    else
    {
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
        return true;
    }
    else
    {
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
    
    Log(Utility::Log_Level::Info, Utility::Log_Sender::API, "Copied image " + std::to_string(idx_image) + " (chain " + std::to_string(idx_chain) + ") to clipboard");
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
        chain->climbing_image.insert(chain->climbing_image.begin() + idx_image, false);
        chain->falling_image.insert(chain->falling_image.begin() + idx_image, false);

		// Add to state
		state->simulation_information_llg[idx_chain].insert(state->simulation_information_llg[idx_chain].begin() + idx_image, std::shared_ptr<Simulation_Information>());

        // Update state
        State_Update(state);

        // Update array lengths
        Chain_Setup_Data(state, idx_chain);

        Log(Utility::Log_Level::Info, Utility::Log_Sender::API, "Inserted image before " + std::to_string(idx_image) + " (chain " + std::to_string(idx_chain) + ") from clipboard");
    }
    else
    {
        Log(Utility::Log_Level::Error, Utility::Log_Sender::API, "Tried to insert image before " + std::to_string(idx_image) + " (chain " + std::to_string(idx_chain) + ") but clipboard was empty");
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
            chain->climbing_image.insert(chain->climbing_image.begin() + idx_image + 1, false);
            chain->falling_image.insert(chain->falling_image.begin() + idx_image + 1, false);
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

        Log(Utility::Log_Level::Info, Utility::Log_Sender::API, "Inserted image after " + std::to_string(idx_image) + " (chain " + std::to_string(idx_chain) + ") from clipboard");
    }
    else
    {
        Log(Utility::Log_Level::Error, Utility::Log_Sender::API, "Tried to insert image after " + std::to_string(idx_image) + " (chain " + std::to_string(idx_chain) + ") but clipboard was empty");
    }
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
        Log(Utility::Log_Level::Info, Utility::Log_Sender::API, "Replaced image " + std::to_string(idx_image) + " (chain " + std::to_string(idx_chain) + ") from clipboard");
    }
    else
    {
        Log(Utility::Log_Level::Error, Utility::Log_Sender::API, "Tried to replace image " + std::to_string(idx_image) + " (chain " + std::to_string(idx_chain) + ") but clipboard was empty");
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
        chain->climbing_image.erase(chain->climbing_image.begin() + idx_image);
        chain->falling_image.erase(chain->falling_image.begin() + idx_image);

		// Remove from state
		state->simulation_information_llg[idx_chain].erase(state->simulation_information_llg[idx_chain].begin() + idx_image);

		// Update State
        State_Update(state);

        // Update array lengths
        Chain_Setup_Data(state, idx_chain);

        Log(Utility::Log_Level::Info, Utility::Log_Sender::API, "Deleted image " + std::to_string(idx_image) + " (chain " + std::to_string(idx_chain) + ")");
        return true;
    }
    else
    {
        Log(Utility::Log_Level::Warning, Utility::Log_Sender::API, "Tried to delete last image (chain " + std::to_string(idx_chain) + ")");
        return false;
    }
}

std::vector<float> Chain_Get_Rx(State * state, int idx_chain_i)
{
	int idx_image = -1, idx_chain = idx_chain_i;
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	// Fetch correct indices and pointers
	from_indices(state, idx_image, idx_chain, image, chain);

	std::vector<float> Rx(chain->Rx.size());
	for (unsigned int i = 0; i < chain->Rx.size(); ++i)
	{
		Rx[i] = (float)chain->Rx[i];
	}
	return Rx;
}

std::vector<float> Chain_Get_Rx_Interpolated(State * state, int idx_chain_i)
{
	int idx_image = -1, idx_chain = idx_chain_i;
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	// Fetch correct indices and pointers
	from_indices(state, idx_image, idx_chain, image, chain);
	
	std::vector<float> Rx_interpolated(chain->Rx_interpolated.size());
	for (unsigned int i = 0; i < chain->Rx_interpolated.size(); ++i)
	{
		Rx_interpolated[i] = (float)chain->Rx_interpolated[i];
	}
	return Rx_interpolated;
}

std::vector<float> Chain_Get_Energy_Interpolated(State * state, int idx_chain_i)
{
	int idx_image = -1, idx_chain = idx_chain_i;
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	// Fetch correct indices and pointers
	from_indices(state, idx_image, idx_chain, image, chain);
	
	std::vector<float> E_interpolated(chain->E_interpolated.size());
	for (unsigned int i = 0; i < chain->E_interpolated.size(); ++i)
	{
		E_interpolated[i] = (float)chain->E_interpolated[i];
	}
	return E_interpolated;
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
        if (i > 0) chain->Rx[i] = chain->Rx[i-1] + Engine::Vectormath::dist_geodesic(*chain->images[i-1]->spins, *chain->images[i]->spins);
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
    chain->Rx_interpolated = std::vector<scalar>((state->noi - 1)*chain->gneb_parameters->n_E_interpolations, 0);
    chain->E_interpolated = std::vector<scalar>((state->noi - 1)*chain->gneb_parameters->n_E_interpolations, 0);
    chain->E_array_interpolated = std::vector<std::vector<scalar>>(7, std::vector<scalar>((state->noi - 1)*chain->gneb_parameters->n_E_interpolations, 0));

    // Initial data update
    Chain_Update_Data(state, idx_chain);
}