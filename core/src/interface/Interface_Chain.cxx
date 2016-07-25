#include "Interface_Chain.h"
#include "Manifoldmath.h"

extern "C" void Chain_next_Image(State * state)
{
    ++state->idx_active_image;
    state->active_image = state->active_chain->images[state->idx_active_image];
    state->active_chain->idx_active_image = state->idx_active_image;
}

extern "C" void Chain_prev_Image(State * state)
{
    --state->idx_active_image;
    state->active_image = state->active_chain->images[state->idx_active_image];
    state->active_chain->idx_active_image = state->idx_active_image;
}

extern "C" void Chain_Insert_Image_Before(State * state, Data::Spin_System & image)
{
    auto c = state->active_chain;
    int idx = state->idx_active_image;
    auto system = std::shared_ptr<Data::Spin_System>(new Data::Spin_System(image));

    c->noi++;
    c->images.insert(c->images.begin() + idx, system);
    
    c->climbing_image.insert(c->climbing_image.begin() + idx, false);
    c->falling_image.insert(c->falling_image.begin() + idx, false);

    state->noi++;
    state->active_image = system;
    state->active_chain->idx_active_image = state->idx_active_image;
}

extern "C" void Chain_Insert_Image_After(State * state, Data::Spin_System & image)
{
    auto c = state->active_chain;
    int idx = state->idx_active_image;
    auto system = std::shared_ptr<Data::Spin_System>(new Data::Spin_System(image));

    if (idx < state->noi - 1) Chain_Insert_Image_Before(state, image);
    else
    {
        c->noi++;
        c->images.push_back(system);
        
        c->climbing_image.push_back(false);
        c->falling_image.push_back(false);
        
        state->noi++;
        state->active_image = system;
        state->active_chain->idx_active_image = state->idx_active_image;
    }
}

extern "C" void Chain_Replace_Image(State * state, Data::Spin_System & image)
{
    auto system = std::shared_ptr<Data::Spin_System>(new Data::Spin_System(image));

    state->active_image = system;
    state->active_chain->idx_active_image = state->idx_active_image;
    state->active_chain->images[state->idx_active_image] = system;
}

extern "C" void Chain_Delete_Image(State * state, int idx)
{
    state->noi--;
    
    state->active_chain->images.erase(state->active_chain->images.begin() + idx);
    state->active_chain->climbing_image.erase(state->active_chain->climbing_image.begin() + idx);
    state->active_chain->falling_image.erase(state->active_chain->falling_image.begin() + idx);

    state->active_image = state->active_chain->images[state->idx_active_image];
    state->active_chain->idx_active_image = state->idx_active_image;
}


extern "C" void Chain_Update_Data(State * state)
{
    auto c = state->active_chain;
    for (int i = 0; i < state->noi; ++i)
    {
        //Engine::Energy::Update(*c->images[i]);
        //c->images[i]->E = c->images[i]->hamiltonian_isotropic->Energy(c->images[i]->spins);
        c->images[i]->UpdateEnergy();
        if (i > 0) c->Rx[i] = c->Rx[i - 1] + Utility::Manifoldmath::Dist_Geodesic(c->images[i - 1]->spins, state->active_chain->images[i]->spins);
    }
}

extern "C" void Chain_Setup_Data(State * state)
{
    auto c = state->active_chain;
    c->Rx = std::vector<double>(state->noi, 0);
    c->Rx_interpolated = std::vector<double>((state->noi - 1)*c->gneb_parameters->n_E_interpolations, 0);
    c->E_interpolated = std::vector<double>((state->noi - 1)*c->gneb_parameters->n_E_interpolations, 0);
    c->E_array_interpolated = std::vector<std::vector<double>>(7, std::vector<double>((state->noi - 1)*c->gneb_parameters->n_E_interpolations, 0));

    c->tangents = std::vector<std::vector<double>>(state->noi, std::vector<double>(3 * state->nos));

    // Initial data update
    Chain_Update_Data(state);
}