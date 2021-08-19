#pragma once
#ifndef SPIRIT_SET_GEOMETRY_HELPERS_HPP
#define SPIRIT_SET_GEOMETRY_HELPERS_HPP

#include <engine/Vectormath.hpp>
#include <Spirit/Simulation.h>

namespace Utility
{

inline void Helper_System_Set_Geometry(std::shared_ptr<Data::Spin_System> system, const Data::Geometry & new_geometry)
{
    // *system->geometry = new_geometry;
    auto old_geometry = *system->geometry;

    // Spins
    int nos_old = system->nos;
    int nos = new_geometry.nos;
    system->nos = nos;

    // Move the vector-fields to the new geometry
    *system->spins = Engine::Vectormath::change_dimensions(
        *system->spins,
        old_geometry.n_cell_atoms, old_geometry.n_cells,
        new_geometry.n_cell_atoms, new_geometry.n_cells,
        {0,0,1});
    system->effective_field = Engine::Vectormath::change_dimensions(
        system->effective_field,
        old_geometry.n_cell_atoms, old_geometry.n_cells,
        new_geometry.n_cell_atoms, new_geometry.n_cells,
        {0,0,0});

    // Update the system geometry
    *system->geometry = new_geometry;

    // Heisenberg Hamiltonian
    if (system->hamiltonian->Name() == "Heisenberg")
        std::static_pointer_cast<Engine::Hamiltonian_Heisenberg>(system->hamiltonian)->Update_Interactions();
}

inline void Helper_State_Set_Geometry(State * state, const Data::Geometry & old_geometry, const Data::Geometry & new_geometry)
{
    // This requires simulations to be stopped, as Methods' temporary arrays may have the wrong size afterwards
    Simulation_Stop_All(state);

    // Lock to avoid memory errors
    state->chain->Lock();
    try
    {
        // Modify all systems in the chain
        for (auto& system : state->chain->images)
        {
            Helper_System_Set_Geometry(system, new_geometry);
        }
    }
    catch( ... )
    {
        spirit_handle_exception_api(-1, -1);
    }
    // Unlock again
    state->chain->Unlock();

    // Retrieve total number of spins
    int nos = state->active_image->nos;

    // Update convenience integerin State
    state->nos = nos;

    // Deal with clipboard image of State
    auto& system = state->clipboard_image;
    if (system)
    {
        // Lock to avoid memory errors
        system->Lock();
        try
        {
            // Modify
            Helper_System_Set_Geometry(system, new_geometry);
        }
        catch( ... )
        {
            spirit_handle_exception_api(-1, -1);
        }
        // Unlock
        system->Unlock();
    }

    // Deal with clipboard configuration of State
    if (state->clipboard_spins)
        *state->clipboard_spins = Engine::Vectormath::change_dimensions(
            *state->clipboard_spins,
            old_geometry.n_cell_atoms, old_geometry.n_cells,
            new_geometry.n_cell_atoms, new_geometry.n_cells,
            {0,0,1});

    // TODO: Deal with Methods
    // for (auto& chain_method_image : state->method_image)
    // {
    //     for (auto& method_image : chain_method_image)
    //     {
    //         method_image->Update_Geometry(new_geometry.n_cell_atoms, new_geometry.n_cells, new_geometry.n_cells);
    //     }
    // }
    // for (auto& method_chain : state->method_chain)
    // {
    //     method_chain->Update_Geometry(new_geometry.n_cell_atoms, new_geometry.n_cells, new_geometry.n_cells);
    // }
}

}
#endif